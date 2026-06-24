// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import DiffusionCore
import ZImageMLX
import AppEngines   // re-exports Flux2DiffusionEngine on macOS only (empty on iOS)

#if os(iOS)
import UIKit
import Photos
#endif

/// The three studio sections (Mac sidebar / iPhone tab bar). Model management lives in Settings
/// and is also reachable from the Create toolbar, so it is not a top-level section.
enum Tab: String, CaseIterable, Identifiable {
    case create, library, settings
    var id: String { rawValue }
    var title: String {
        switch self { case .create: "Create"; case .library: "Library"; case .settings: "Settings" }
    }
    var icon: String {
        switch self {
        case .create: "wand.and.stars"
        case .library: "photo.on.rectangle.angled"; case .settings: "gearshape"
        }
    }
}

/// In-app appearance override. `.system` follows the device; the others force a scheme.
enum AppTheme: String, CaseIterable, Identifiable {
    case system, light, dark
    var id: String { rawValue }
    var label: String {
        switch self { case .system: "System"; case .light: "Light"; case .dark: "Dark" }
    }
    var colorScheme: ColorScheme? {
        switch self { case .system: nil; case .light: .light; case .dark: .dark }
    }
}

/// One finished generation, kept in the Library (in-session for now).
struct Generation: Identifiable {
    let id = UUID()
    let image: CGImage
    let prompt: String
    let modelID: String
    let modelName: String
    let size: Int
    let steps: Int
    let seed: UInt64
    let duration: TimeInterval
    let date = Date()
}

/// Format a generation time for display: "12.4s" under a minute, "1m 05s" above.
func formatDuration(_ seconds: TimeInterval) -> String {
    if seconds < 60 { return String(format: "%.1fs", seconds) }
    let m = Int(seconds) / 60, s = Int(seconds) % 60
    return "\(m)m \(String(format: "%02d", s))s"
}

/// Drives any catalog model through a `DiffusionEngine` facade (Z-Image and — on macOS — FLUX.2),
/// with model switching. UI state lives on the main actor; the engines are actors, so their heavy
/// MLX work runs off-main without blocking the UI. Z-Image weights are downloaded in-app first;
/// FLUX self-downloads inside its `load`.
@MainActor
@Observable
final class AppModel {
    enum Phase: Equatable {
        case idle
        case downloading(Double)
        case loading(Double)
        case generating(Int, Int)
        case done
        case failed(String)
    }

    enum AppError: LocalizedError {
        case unsupportedOnPlatform(String)
        var errorDescription: String? {
            switch self { case .unsupportedOnPlatform(let m): return m }
        }
    }

    let models = Catalog.all
    let device = DeviceTier.current
    var tab: Tab = .create
    var selectedID: String = Catalog.all.first!.id
    var prompt = "a red panda on a mossy rock, soft morning light"
    var size = 1024
    var steps = 8
    var seedText = "42"
    var phase: Phase = .idle
    var image: CGImage?
    var history: [Generation] = []
    /// Peak resident memory (phys_footprint, the value jetsam checks) seen during the last
    /// generation — surfaced on iPhone so the streaming residency is visible. 0 until a run happens.
    var peakResidentBytes: UInt64 = 0
    /// Wall-clock time of the last completed generation (encode → denoise → decode), shown in the
    /// status line and recorded with each Library entry.
    var lastGenerationSeconds: TimeInterval?
    /// Bumped after any model/component download or delete so views re-read on-disk install state.
    private(set) var componentsRevision = 0

    /// In-app appearance override, persisted across launches (defaults to following the system).
    var appearance: AppTheme = .system {
        didSet { UserDefaults.standard.set(appearance.rawValue, forKey: "appearance") }
    }

    #if os(macOS)
    /// FLUX precision preferences (macOS only — FLUX is macOS-only), persisted across launches.
    var fluxTransformer: Flux2FacadeEngine.FluxTransformerPrecision = .bit8 {
        didSet { UserDefaults.standard.set(fluxTransformer.rawValue, forKey: "fluxTransformer") }
    }
    var fluxEncoder: Flux2FacadeEngine.FluxEncoderPrecision = .bit8 {
        didSet { UserDefaults.standard.set(fluxEncoder.rawValue, forKey: "fluxEncoder") }
    }

    /// Which FLUX component (by id) is downloading + its 0...1 progress, for the detail's list.
    var fluxComponentDownloadID: String?
    var fluxComponentFraction: Double = 0
    /// A failed component download, surfaced as inline Retry — kept off the shared generation `phase`
    /// so a failed Get never leaves a sticky "Failed" on the Create canvas.
    var fluxComponentError: (id: String, message: String)?

    static func friendlyDownloadError(_ error: Error) -> String {
        if let urlError = error as? URLError, urlError.code == .notConnectedToInternet { return "No connection" }
        return "Download failed"
    }

    /// The individually-managed FLUX components and their current on-disk state.
    func fluxComponents() -> [Flux2FacadeEngine.Flux2ComponentInfo] { Flux2FacadeEngine.allComponents() }

    /// Download one FLUX component by id (one at a time; shares the download/generate lock).
    func downloadFluxComponent(_ id: String) async {
        guard !inFlight else { return }
        inFlight = true; defer { inFlight = false }
        fluxComponentError = nil
        fluxComponentDownloadID = id; fluxComponentFraction = 0
        defer { fluxComponentDownloadID = nil; componentsRevision += 1 }
        do {
            try await Flux2FacadeEngine.downloadComponent(id) { fraction in
                // Ignore a stale callback from a previous download.
                Task { @MainActor in if self.fluxComponentDownloadID == id { self.fluxComponentFraction = fraction } }
            }
        } catch {
            fluxComponentError = (id, Self.friendlyDownloadError(error))
        }
    }

    /// Delete one FLUX component's weights by id. If it's part of the loaded recipe, unload first
    /// so generation never runs against missing weights.
    func deleteFluxComponent(_ id: String) async {
        guard !inFlight else { return }
        inFlight = true; defer { inFlight = false }
        if fluxActiveComponentIDs.contains(id),
           let loaded = models.first(where: { $0.id == loadedID }), loaded.family == .flux2,
           let current = engine {
            engine = nil; loadedID = nil; loadedRecipe = nil
            await current.unload()
        }
        try? Flux2FacadeEngine.deleteComponent(id)
        componentsRevision += 1
    }

    /// The component ids the currently-selected FLUX precision actually runs ("active recipe").
    var fluxActiveComponentIDs: [String] {
        Flux2FacadeEngine.activeComponentIDs(transformer: fluxTransformer, encoder: fluxEncoder)
    }

    /// Total download size of the active FLUX precision (for the card's size chip).
    var fluxActiveBytes: Int64 {
        let active = Set(fluxActiveComponentIDs)
        return fluxComponents().filter { active.contains($0.id) }.reduce(0) { $0 + $1.bytes }
    }

    /// Short label of the active FLUX recipe, e.g. "8-bit · 4-bit encoder".
    var fluxRecipeLabel: String { "\(fluxTransformer.label) · \(fluxEncoder.label) encoder" }

    /// Active-recipe components not yet on disk (drives the quantified Download label + tri-state).
    var fluxActiveMissing: [Flux2FacadeEngine.Flux2ComponentInfo] {
        let active = Set(fluxActiveComponentIDs)
        return fluxComponents().filter { active.contains($0.id) && !$0.isDownloaded }
    }
    var fluxMissingBytes: Int64 { fluxActiveMissing.reduce(0) { $0 + $1.bytes } }
    var fluxMissingCount: Int { fluxActiveMissing.count }
    var fluxActiveCount: Int { fluxActiveComponentIDs.count }

    /// "Complete · 570 MB" when partly installed, else "Download · 4.6 GB" — so the action says
    /// exactly what one tap fetches.
    var fluxDownloadActionLabel: String {
        let bytes = ByteCountFormatter.string(fromByteCount: fluxMissingBytes, countStyle: .file)
        return (fluxMissingCount < fluxActiveCount) ? "Complete · \(bytes)" : "Download · \(bytes)"
    }
    #endif

    private let downloader: ModelDownloader
    private var engine: (any DiffusionEngine)?
    private var loadedID: String?
    private var loadedRecipe: String?   // FLUX: the active recipe label that's loaded, for reload-on-precision-change
    private var inFlight = false   // reentrancy lock: one download/generate at a time

    init() {
        let support = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let base = (support ?? URL(fileURLWithPath: NSTemporaryDirectory())).appending(component: "MobileDiffuser")
        downloader = ModelDownloader(downloadBase: base)
        if let raw = UserDefaults.standard.string(forKey: "appearance"), let theme = AppTheme(rawValue: raw) {
            appearance = theme   // set in init: didSet does not fire, so no redundant write-back
        }
        #if os(macOS)
        if let raw = UserDefaults.standard.string(forKey: "fluxTransformer"),
           let value = Flux2FacadeEngine.FluxTransformerPrecision(rawValue: raw) { fluxTransformer = value }
        if let raw = UserDefaults.standard.string(forKey: "fluxEncoder"),
           let value = Flux2FacadeEngine.FluxEncoderPrecision(rawValue: raw) { fluxEncoder = value }
        #endif
    }

    var selected: DiffusionModel { models.first { $0.id == selectedID } ?? models[0] }

    /// Where downloaded models live (shown in Settings).
    var storageLocation: String { downloader.downloadBase.appending(component: "models").path }

    /// In-app download applies to Z-Image; FLUX manages its own weights inside `load`.
    var managesOwnDownload: Bool { selected.family != .zImage }

    var isDownloaded: Bool { isDownloaded(selected) }

    var isBusy: Bool {
        switch phase { case .downloading, .loading, .generating: return true; default: break }
        #if os(macOS)
        if fluxComponentDownloadID != nil { return true }   // a per-component install is in flight
        #endif
        return false
    }
    var isFailed: Bool { if case .failed = phase { return true } else { return false } }

    var statusText: String {
        switch phase {
        case .idle: return managesOwnDownload ? "Ready to load" : (isDownloaded ? "Ready" : "Not downloaded")
        case .downloading(let f): return "Downloading… \(Int(f * 100))%"
        case .loading(let f): return "Loading into memory… \(Int(f * 100))%"
        case .generating(let s, let t): return "Generating… step \(s)/\(t)"
        case .done: return lastGenerationSeconds.map { "Done in \(formatDuration($0))" } ?? "Done"
        case .failed(let m): return "Failed: \(m)"
        }
    }

    /// Peak memory of the last generation vs the device budget — shown on iPhone to make the
    /// streaming residency visible (does the 6B model stay under the jetsam ceiling?).
    var memoryReadout: String? {
        guard peakResidentBytes > 0 else { return nil }
        let peak = Double(peakResidentBytes) / 1_073_741_824
        let budget = Double(device.memoryBudgetBytes) / 1_073_741_824
        return String(format: "peak %.1f / %.1f GB", peak, budget)
    }
    func download() async {
        guard !inFlight else { return }
        inFlight = true; defer { inFlight = false }
        await downloadSelected()
        componentsRevision += 1   // model-level download changed the on-disk component set
    }

    func generate() async {
        guard !inFlight else { return }
        inFlight = true; defer { inFlight = false }
        await runGenerate()
    }

    /// Delete a model's downloaded weights to free disk space. If that model is currently loaded,
    /// it is unloaded first so memory is freed too.
    func delete(_ model: DiffusionModel) async {
        guard !inFlight else { return }
        inFlight = true; defer { inFlight = false }
        if loadedID == model.id, let current = engine {
            engine = nil; loadedID = nil; loadedRecipe = nil
            await current.unload()
        }
        switch model.family {
        case .zImage:
            let url = downloader.localURL(repoId: model.variants[0].source.huggingFaceRepo)
            try? FileManager.default.removeItem(at: url)
        case .flux2:
            #if os(macOS)
            try? Flux2FacadeEngine.deleteWeights()
            #endif
        default:
            break
        }
        phase = .idle
        componentsRevision += 1   // model-level delete changed the on-disk component set

    }

    // MARK: - Unified model recipe (one UI-facing shape for every family)

    /// Build a `ModelRecipe` for a model — its components (on-disk + active state) and precision axes —
    /// so the detail screen renders one data-driven template regardless of family.
    func recipe(for model: DiffusionModel) -> ModelRecipe {
        switch model.family {
        case .zImage:
            let v = model.variants[0]
            let comp = RecipeComponent(
                id: "zimage", title: "\(model.displayName) · \(v.precision.label)",
                subtitle: model.summary, kind: .weights,
                repo: v.source.huggingFaceRepo, bytes: v.approximateBytes,
                isDownloaded: isDownloaded(model), isActive: true)
            return ModelRecipe(axes: [], components: [comp])
        case .flux2:
            #if os(macOS)
            _ = componentsRevision   // re-read the on-disk list when it changes
            let active = Set(fluxActiveComponentIDs)
            let comps: [RecipeComponent] = fluxComponents().map { c in
                let kind: RecipeComponent.Kind = {
                    switch c.kind { case .transformer: return .transformer
                    case .textEncoder: return .textEncoder; case .vae: return .vae }
                }()
                return RecipeComponent(id: c.id, title: c.title, subtitle: c.subtitle, kind: kind,
                                       repo: c.repo, bytes: c.bytes, isDownloaded: c.isDownloaded,
                                       isActive: active.contains(c.id))
            }
            let axes: [PrecisionAxis] = [
                PrecisionAxis(id: "transformer", title: "Model precision",
                    options: Flux2FacadeEngine.FluxTransformerPrecision.allCases.map {
                        PrecisionOption(id: $0.rawValue, label: $0.label, note: $0.note) },
                    selectedID: fluxTransformer.rawValue),
                PrecisionAxis(id: "encoder", title: "Text encoder",
                    options: Flux2FacadeEngine.FluxEncoderPrecision.allCases.map {
                        PrecisionOption(id: $0.rawValue, label: $0.label, note: $0.note) },
                    selectedID: fluxEncoder.rawValue),
            ]
            return ModelRecipe(axes: axes, components: comps)
            #else
            return ModelRecipe()
            #endif
        default:
            return ModelRecipe()
        }
    }

    /// Apply a precision-axis choice (FLUX only today).
    func setPrecision(axisID: String, optionID: String) {
        #if os(macOS)
        if axisID == "transformer", let v = Flux2FacadeEngine.FluxTransformerPrecision(rawValue: optionID) { fluxTransformer = v }
        if axisID == "encoder", let v = Flux2FacadeEngine.FluxEncoderPrecision(rawValue: optionID) { fluxEncoder = v }
        #endif
    }

    /// 0...1 progress for a component currently installing (nil if it isn't).
    func componentProgress(_ id: String, model: DiffusionModel) -> Double? {
        #if os(macOS)
        if model.family == .flux2, fluxComponentDownloadID == id { return fluxComponentFraction }
        #endif
        if model.family == .zImage, id == "zimage", selectedID == model.id,
           case .downloading(let f) = phase { return f }
        return nil
    }

    /// A friendly error for a component whose last install failed (nil otherwise).
    func componentErrorMessage(_ id: String) -> String? {
        #if os(macOS)
        if fluxComponentError?.id == id { return fluxComponentError?.message }
        #endif
        return nil
    }

    /// Install one recipe component.
    func installComponent(_ id: String, model: DiffusionModel) async {
        #if os(macOS)
        if model.family == .flux2 { await downloadFluxComponent(id); return }
        #endif
        if model.family == .zImage { selectedID = model.id; await download() }
    }

    /// Remove one recipe component.
    func removeComponent(_ id: String, model: DiffusionModel) async {
        #if os(macOS)
        if model.family == .flux2 { await deleteFluxComponent(id); return }
        #endif
        if model.family == .zImage { await delete(model) }
    }

    /// Download all missing active components for a model (the model-level "Download" action).
    func installRecipe(_ model: DiffusionModel) async {
        selectedID = model.id
        await download()
    }

    /// Download the selected model's weights. No reentrancy guard — callers hold `inFlight`.
    private func downloadSelected() async {
        let model = selected
        #if os(iOS)
        // Keep the screen awake during the multi-GB download so a foreground URLSession isn't
        // killed by auto-lock mid-transfer.
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = false }
        #endif
        do {
            phase = .downloading(0)
            switch model.family {
            case .zImage:
                let repo = model.variants[0].source.huggingFaceRepo
                _ = try await downloader.download(repoId: repo) { fraction in
                    Task { @MainActor in if case .downloading = self.phase { self.phase = .downloading(fraction) } }
                }
            case .flux2:
                #if os(macOS)
                try await Flux2FacadeEngine.download(transformer: fluxTransformer, encoder: fluxEncoder) { fraction in
                    Task { @MainActor in if case .downloading = self.phase { self.phase = .downloading(fraction) } }
                }
                #endif
            default:
                break
            }
            phase = .idle
        } catch {
            phase = .failed(String(describing: error))
        }
    }

    private func runGenerate() async {
        let model = selected
        do {
            // Ensure the selected model's active recipe is on disk before loading (all families).
            if !isDownloaded(model) {
                await downloadSelected()
                guard isDownloaded(model) else { return }
            }

            // Reload when the model changed OR (FLUX) the chosen precision recipe changed.
            var needsReload = engine == nil || loadedID != model.id
            #if os(macOS)
            if model.family == .flux2, loadedRecipe != fluxRecipeLabel { needsReload = true }
            #endif
            if needsReload {
                // Unload the previous engine BEFORE loading the new one, so two large weight sets
                // are never resident at once (a model switch would otherwise peak ~10+ GB).
                if let previous = engine {
                    engine = nil; loadedID = nil; loadedRecipe = nil
                    await previous.unload()
                }
                let built = try makeEngine(for: model)
                // The streaming Z-Image engine needs the real per-component source (it reads weights
                // through it); the resident facades ignore `source` and self-resolve their weights,
                // so they get a cheap empty placeholder.
                let source: WeightSource = (model.family == .zImage && zImageUsesStreaming)
                    ? try zImageSource(for: model, streaming: true)
                    : SafetensorsWeightSource(tensors: [:])
                phase = .loading(0)
                try await built.load(model, variant: model.variants[0], source: source) { fraction in
                    Task { @MainActor in if case .loading = self.phase { self.phase = .loading(fraction) } }
                }
                engine = built
                loadedID = model.id
                #if os(macOS)
                loadedRecipe = (model.family == .flux2) ? fluxRecipeLabel : nil
                #endif
            }
            // Ensure the loaded engine is actually the selected model (a failed switch leaves none).
            guard let engine, loadedID == model.id else { phase = .failed("Model not loaded"); return }

            phase = .generating(0, steps)
            // Sample peak resident memory across the run so the iPhone streaming residency is visible.
            peakResidentBytes = 0
            let memoryMonitor = Task { @MainActor in
                while !Task.isCancelled {
                    self.peakResidentBytes = max(self.peakResidentBytes, MemoryProbe.residentBytes())
                    try? await Task.sleep(nanoseconds: 300_000_000)
                }
            }
            defer { memoryMonitor.cancel() }
            let seed = UInt64(seedText) ?? 42
            let request = GenerationRequest(prompt: prompt, steps: steps, seed: seed,
                                            size: ImageSize(width: size, height: size))
            let genStart = Date()
            let cgImage = try await engine.generate(request) { progress in
                Task { @MainActor in
                    // Only advance while still generating, so a late callback can't revive a finished run.
                    if case .denoising(let step, let total, _) = progress, case .generating = self.phase {
                        self.phase = .generating(step, total)
                    }
                }
            }
            let elapsed = Date().timeIntervalSince(genStart)
            lastGenerationSeconds = elapsed
            image = cgImage
            phase = .done
            history.insert(Generation(image: cgImage, prompt: prompt, modelID: model.id,
                                      modelName: model.displayName, size: size, steps: steps, seed: seed,
                                      duration: elapsed),
                           at: 0)
        } catch {
            phase = .failed(String(describing: error))
        }
    }

    /// Hardware-fit for a catalog model on this device (drives the gallery's fit badges).
    func capabilities(for model: DiffusionModel) -> EngineCapabilities {
        let variant = model.variants[0]
        switch model.family {
        case .zImage:
            // The fit badge must match what actually runs. On iPhone Z-Image runs through the
            // block-streaming MLXDiffusionEngine (residency .streamingInternal, ~3 GB peak), so report
            // that plan — not the resident facade's ~7.9 GB peak, which would show "unsupported".
            // Mac stays on the resident facade (the path that runs there).
            if zImageUsesStreaming {
                return MLXDiffusionEngine.capabilities(for: model, variant: variant, on: device)
            }
            return ZImageFacadeEngine.capabilities(for: model, variant: variant, on: device)
        case .flux2:
            #if os(macOS)
            return Flux2FacadeEngine.capabilities(for: model, variant: variant, on: device)
            #else
            return EngineCapabilities(runnable: false, residency: .unsupported,
                                      estimatedPeakBytes: variant.approximateBytes, note: "macOS only")
            #endif
        default:
            return EngineCapabilities(runnable: false, residency: .unsupported,
                                      estimatedPeakBytes: variant.approximateBytes, note: "Unsupported")
        }
    }

    func isDownloaded(_ model: DiffusionModel) -> Bool {
        switch model.family {
        case .zImage:
            return downloader.isDownloaded(repoId: model.variants[0].source.huggingFaceRepo)
        case .flux2:
            // FLUX self-manages its weights inside the engine; ask it whether the chosen precision
            // (transformer + matching Qwen3 encoder + VAE) is on disk.
            #if os(macOS)
            return Flux2FacadeEngine.isDownloaded(transformer: fluxTransformer, encoder: fluxEncoder)
            #else
            return false
            #endif
        default:
            return false
        }
    }

    /// How many catalog models have weights on disk (shown in Settings).
    var downloadedModelCount: Int { models.filter { isDownloaded($0) }.count }

    /// Total on-disk size of downloaded model weights, walked off the main actor.
    func storageUsedBytes() async -> Int64 {
        let dir = downloader.downloadBase.appending(component: "models")
        return await Task.detached {
            var total: Int64 = 0
            if let walker = FileManager.default.enumerator(
                at: dir, includingPropertiesForKeys: [.totalFileAllocatedSizeKey]) {
                for case let url as URL in walker {
                    let size = (try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey]))?.totalFileAllocatedSize
                    total += Int64(size ?? 0)
                }
            }
            #if os(macOS)
            total += Flux2FacadeEngine.downloadedBytes()   // FLUX weights live in the engine's own cache
            #endif
            return total
        }.value
    }

    /// Apply a past generation's settings and jump to Create.
    func reuse(_ g: Generation) {
        prompt = g.prompt; size = g.size; steps = g.steps; seedText = String(g.seed)
        if models.contains(where: { $0.id == g.modelID }) { selectedID = g.modelID }
        tab = .create
    }

    /// Z-Image runs the block-streaming `MLXDiffusionEngine` on iPhone (partial-load, ~3 GB peak)
    /// and stays on the resident `ZImageFacadeEngine` on Mac (it already produces correct images).
    private var zImageUsesStreaming: Bool { device.isPhone }

    /// Build the `WeightSource` a streaming Z-Image engine consumes: a `ZImageComponentSource`
    /// opened over the downloaded model folder. `streaming` opens the transformer via
    /// `RangedFileWeightSource` (pread on demand, frees on release). The resident facade ignores
    /// `source`, so this is only built for the streaming path.
    private func zImageSource(for model: DiffusionModel, streaming: Bool) throws -> WeightSource {
        let dir = downloader.localURL(repoId: model.variants[0].source.huggingFaceRepo)
        return try ZImageComponentSource.open(modelDirectory: dir, streaming: streaming)
    }

    private func makeEngine(for model: DiffusionModel) throws -> any DiffusionEngine {
        switch model.family {
        case .zImage:
            if zImageUsesStreaming {
                // iPhone partial-load path: the generic MLX engine drives ZImageArchitecture,
                // loading/releasing each transformer block per step from the streaming source.
                return MLXDiffusionEngine(architecture: ZImageArchitecture(), device: device)
            }
            let dir = downloader.localURL(repoId: model.variants[0].source.huggingFaceRepo)
            return ZImageFacadeEngine(modelDirectory: dir)
        case .flux2:
            #if os(macOS)
            return Flux2FacadeEngine(transformer: fluxTransformer, encoder: fluxEncoder)
            #else
            throw AppError.unsupportedOnPlatform("FLUX.2 runs on macOS only.")
            #endif
        default:
            throw AppError.unsupportedOnPlatform("\(model.displayName) is not supported yet.")
        }
    }

    // MARK: - Image export

    /// PNG bytes for a `CGImage`, used by the macOS save panel / `.fileExporter`. Cross-platform.
    func pngData(_ cg: CGImage) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }

    #if os(iOS)
    /// Save a generated image to the user's Photo library (add-only authorization). Denial is
    /// non-fatal — the image stays in the in-app Library. iOS only; macOS uses a save panel in-view.
    func exportImage(_ cg: CGImage) {
        let image = UIImage(cgImage: cg)
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else { return }
            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            } completionHandler: { _, _ in }
        }
    }
    #endif
}
