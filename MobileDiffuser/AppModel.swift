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
    let date = Date()
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
    /// Bumped after a component download/delete so views re-read the on-disk component list.
    private(set) var componentsRevision = 0

    /// The individually-managed FLUX components and their current on-disk state.
    func fluxComponents() -> [Flux2FacadeEngine.Flux2ComponentInfo] { Flux2FacadeEngine.allComponents() }

    /// Download one FLUX component by id (one at a time; shares the download/generate lock).
    func downloadFluxComponent(_ id: String) async {
        guard !inFlight else { return }
        inFlight = true; defer { inFlight = false }
        fluxComponentDownloadID = id; fluxComponentFraction = 0
        defer { fluxComponentDownloadID = nil; componentsRevision += 1 }
        do {
            try await Flux2FacadeEngine.downloadComponent(id) { fraction in
                Task { @MainActor in self.fluxComponentFraction = fraction }
            }
        } catch {
            phase = .failed(String(describing: error))
        }
    }

    /// Delete one FLUX component's weights by id.
    func deleteFluxComponent(_ id: String) {
        guard !inFlight else { return }
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
        switch phase { case .downloading, .loading, .generating: return true; default: return false }
    }
    var isFailed: Bool { if case .failed = phase { return true } else { return false } }

    var statusText: String {
        switch phase {
        case .idle: return managesOwnDownload ? "Ready to load" : (isDownloaded ? "Ready" : "Not downloaded")
        case .downloading(let f): return "Downloading… \(Int(f * 100))%"
        case .loading(let f): return "Loading into memory… \(Int(f * 100))%"
        case .generating(let s, let t): return "Generating… step \(s)/\(t)"
        case .done: return "Done"
        case .failed(let m): return "Failed: \(m)"
        }
    }

    /// Explicit download of the selected Z-Image model (no-op for self-managing families).
    func download() async {
        guard !inFlight else { return }
        inFlight = true; defer { inFlight = false }
        await downloadSelected()
        #if os(macOS)
        componentsRevision += 1   // model-level download changed the on-disk component set
        #endif
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
            engine = nil; loadedID = nil
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
        #if os(macOS)
        componentsRevision += 1   // model-level delete changed the on-disk component set
        #endif
    }

    /// Download the selected model's weights. No reentrancy guard — callers hold `inFlight`.
    private func downloadSelected() async {
        let model = selected
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
            // Z-Image needs its weights on disk before the engine can load them.
            if model.family == .zImage, !isDownloaded {
                await downloadSelected()
                guard isDownloaded else { return }
            }

            if engine == nil || loadedID != model.id {
                // Unload the previous engine BEFORE loading the new one, so two large weight sets
                // are never resident at once (a model switch would otherwise peak ~10+ GB).
                if let previous = engine {
                    engine = nil; loadedID = nil
                    await previous.unload()
                }
                let built = try makeEngine(for: model)
                phase = .loading(0)
                try await built.load(model, variant: model.variants[0], source: SafetensorsWeightSource(tensors: [:])) { fraction in
                    Task { @MainActor in if case .loading = self.phase { self.phase = .loading(fraction) } }
                }
                engine = built
                loadedID = model.id
            }
            // Ensure the loaded engine is actually the selected model (a failed switch leaves none).
            guard let engine, loadedID == model.id else { phase = .failed("Model not loaded"); return }

            phase = .generating(0, steps)
            let seed = UInt64(seedText) ?? 42
            let request = GenerationRequest(prompt: prompt, steps: steps, seed: seed,
                                            size: ImageSize(width: size, height: size))
            let cgImage = try await engine.generate(request) { progress in
                Task { @MainActor in
                    // Only advance while still generating, so a late callback can't revive a finished run.
                    if case .denoising(let step, let total, _) = progress, case .generating = self.phase {
                        self.phase = .generating(step, total)
                    }
                }
            }
            image = cgImage
            phase = .done
            history.insert(Generation(image: cgImage, prompt: prompt, modelID: model.id,
                                      modelName: model.displayName, size: size, steps: steps, seed: seed),
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

    private func makeEngine(for model: DiffusionModel) throws -> any DiffusionEngine {
        switch model.family {
        case .zImage:
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
