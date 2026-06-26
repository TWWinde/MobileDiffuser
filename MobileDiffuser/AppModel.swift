// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import DiffusionCore
import ZImageMLX
import AppEngines   // re-exports the cross-platform Flux2DiffusionEngine facade

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

/// One finished generation, kept in the durable local Library.
struct Generation: Identifiable, @unchecked Sendable {
    let id: UUID
    let image: CGImage
    let prompt: String
    let modelID: String
    let modelName: String
    let size: Int
    let steps: Int
    let seed: UInt64
    let duration: TimeInterval
    /// Model-specific recipe rows captured at generation time (FLUX: transformer/encoder/decoder;
    /// Z-Image: precision; future families: their own). Generic so the Library detail can show whatever
    /// settings a model has without hardcoding any one model's fields onto `Generation`.
    let settings: [GenerationSetting]
    let date: Date
    /// PNG filename under Library/images. New records default to `<uuid>.png`.
    let imageFilename: String?

    init(id: UUID = UUID(), image: CGImage, prompt: String, modelID: String, modelName: String,
         size: Int, steps: Int, seed: UInt64, duration: TimeInterval,
         settings: [GenerationSetting], date: Date = Date(), imageFilename: String? = nil) {
        self.id = id
        self.image = image
        self.prompt = prompt
        self.modelID = modelID
        self.modelName = modelName
        self.size = size
        self.steps = steps
        self.seed = seed
        self.duration = duration
        self.settings = settings
        self.date = date
        self.imageFilename = imageFilename
    }
}

/// One model-specific setting recorded with a generation — shown in the Library detail, and (when it
/// carries an axis) restorable by "Reuse settings". Decoupling these from `Generation`'s fixed fields
/// is what lets each model family contribute different settings, now and in the future.
struct GenerationSetting: Identifiable, Hashable, Codable, Sendable {
    var id: String { label }
    let label: String       // e.g. "Decoder"
    let value: String       // e.g. "Standard VAE"
    /// The recipe axis id + option rawValue this came from, so Reuse can restore the exact recipe.
    /// `nil` for informational rows that aren't a selectable axis (e.g. Z-Image's fixed precision).
    var axisID: String? = nil
    var optionID: String? = nil
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
        case pausing(Int, Int)
        case paused(Int, Int)
        /// Auto-paused to let the device cool. NOT a failure — it resumes on its own once the phone
        /// cools. Carries the step it paused at so the UI keeps the progress context.
        case cooling(Int, Int)
        case cancelling
        case cancelled
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
    var selectedID: String = Catalog.all.first!.id {
        didSet {
            guard selectedID != oldValue else { return }
            UserDefaults.standard.set(selectedID, forKey: "selectedModelID")   // survive relaunch
            applyModelDefaults()   // each model has its own native step count + size
        }
    }
    var prompt = "a red panda on a mossy rock, soft morning light"
    /// Default render size: 512 on iPhone (a 1024 latent's attention + VAE-decode working set risks
    /// jetsam on a phone), 1024 on Mac. Overridden in `init()` once `device` is known.
    var size = 1024 {
        didSet {
            #if os(iOS)
            // The standard VAE's wider decoder channels are only memory-safe up to 512 on a phone;
            // if the size is bumped past it, fall back to the small decoder so decode can't OOM.
            if size > 512, fluxDecoder == .standard { fluxDecoder = .small }
            #endif
        }
    }
    var steps = 8
    var seedText = "42"
    var phase: Phase = .idle
    var image: CGImage?
    var history: [Generation] = []
    /// Transient confirmation banner (e.g. "Saved to Photos"); auto-clears after a couple seconds.
    var toast: String?
    @ObservationIgnored private var toastTask: Task<Void, Never>?
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

    /// Default FLUX precision by device: iPhone defaults to the pre-quantized 4-bit transformer +
    /// 4-bit encoder (the only recipe that fits the phone's memory budget; loads with no spike);
    /// Mac defaults to 8-bit. The persisted-pref keys are shared, so saved Mac choices survive.
    #if os(iOS)
    static let defaultFluxTransformer: Flux2FacadeEngine.FluxTransformerPrecision = .bit4
    static let defaultFluxEncoder: Flux2FacadeEngine.FluxEncoderPrecision = .bit4
    #else
    static let defaultFluxTransformer: Flux2FacadeEngine.FluxTransformerPrecision = .bit8
    static let defaultFluxEncoder: Flux2FacadeEngine.FluxEncoderPrecision = .bit8
    #endif

    /// Decoder defaults to the small VAE on both platforms — it's the lightest, fits iPhone at any
    /// size, and is Apache-2.0 (the standard VAE is sharper but FLUX.2 Non-Commercial). Users opt in.
    static let defaultFluxDecoder: Flux2FacadeEngine.FluxDecoderPrecision = .small

    /// FLUX precision preferences, persisted across launches.
    var fluxTransformer: Flux2FacadeEngine.FluxTransformerPrecision = AppModel.defaultFluxTransformer {
        didSet { UserDefaults.standard.set(fluxTransformer.rawValue, forKey: "fluxTransformer") }
    }
    var fluxEncoder: Flux2FacadeEngine.FluxEncoderPrecision = AppModel.defaultFluxEncoder {
        didSet { UserDefaults.standard.set(fluxEncoder.rawValue, forKey: "fluxEncoder") }
    }
    var fluxDecoder: Flux2FacadeEngine.FluxDecoderPrecision = AppModel.defaultFluxDecoder {
        didSet { UserDefaults.standard.set(fluxDecoder.rawValue, forKey: "fluxDecoder") }
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
        let operationID = UUID()
        activeOperationID = operationID
        inFlight = true; defer { inFlight = false; activeOperationID = nil }
        await downloadFluxComponentUnlocked(id, operationID: operationID)
    }

    private func downloadFluxComponentUnlocked(_ id: String, operationID: UUID) async {
        fluxComponentError = nil
        fluxComponentDownloadID = id; fluxComponentFraction = 0
        defer { fluxComponentDownloadID = nil; componentsRevision += 1 }
        do {
            try await Flux2FacadeEngine.downloadComponent(id) { fraction in
                // Ignore a stale callback from a previous download.
                Task { @MainActor in
                    if self.activeOperationID == operationID, self.fluxComponentDownloadID == id {
                        self.fluxComponentFraction = fraction
                    }
                }
            }
        } catch is CancellationError {
            phase = .cancelled
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
            engine = nil; loadedID = nil; loadedRecipe = nil; loadedFluxStreaming = nil
            await current.unload()
        }
        try? Flux2FacadeEngine.deleteComponent(id)
        componentsRevision += 1
    }

    /// The component ids the currently-selected FLUX precision actually runs ("active recipe").
    var fluxActiveComponentIDs: [String] {
        Flux2FacadeEngine.activeComponentIDs(transformer: fluxTransformer, encoder: fluxEncoder, decoder: fluxDecoder)
    }

    /// Total download size of the active FLUX precision (for the card's size chip).
    var fluxActiveBytes: Int64 {
        let active = Set(fluxActiveComponentIDs)
        return fluxComponents().filter { active.contains($0.id) }.reduce(0) { $0 + $1.bytes }
    }

    /// Short label of the active FLUX recipe, e.g. "8-bit · 4-bit encoder · Small decoder".
    /// NOTE: this is the `loadedRecipe` reload key — the decoder MUST be included so switching
    /// decoders forces a pipeline reload (otherwise generation would decode with the stale VAE).
    var fluxRecipeLabel: String {
        "\(fluxTransformer.label) · \(fluxEncoder.label) encoder · \(fluxDecoder.label)"
    }

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

    private let downloader: ModelDownloader
    @ObservationIgnored private let libraryStore: LibraryStore
    private var engine: (any DiffusionEngine)?
    private var loadedID: String?
    private var loadedRecipe: String?   // FLUX: the active recipe label that's loaded, for reload-on-precision-change
    private var loadedFluxStreaming: Bool?   // FLUX: whether the loaded engine is the streaming (1024) one, for reload-on-size-cross
    private var inFlight = false   // reentrancy lock: one download/generate at a time
    @ObservationIgnored private var generationTask: Task<Void, Never>?
    @ObservationIgnored private var operationTask: Task<Void, Never>?
    @ObservationIgnored private var generationControl: GenerationControl?
    @ObservationIgnored private var activeOperationID: UUID?

    init() {
        let support = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let base = (support ?? URL(fileURLWithPath: NSTemporaryDirectory())).appending(component: "MobileDiffuser")
        downloader = ModelDownloader(downloadBase: base)
        libraryStore = LibraryStore(baseURL: base)
        // Restore the last-selected model BEFORE applying its defaults. An assignment inside init does
        // not fire the `didSet`, so set it directly and let the explicit applyModelDefaults() below pick
        // up the restored selection (validate it still exists in the catalog).
        if let raw = UserDefaults.standard.string(forKey: "selectedModelID"),
           models.contains(where: { $0.id == raw }) {
            selectedID = raw
        }
        applyModelDefaults()   // steps + size default to the selected model's native values
        if let raw = UserDefaults.standard.string(forKey: "appearance"), let theme = AppTheme(rawValue: raw) {
            appearance = theme   // set in init: didSet does not fire, so no redundant write-back
        }
        if let raw = UserDefaults.standard.string(forKey: "fluxTransformer"),
           let value = Flux2FacadeEngine.FluxTransformerPrecision(rawValue: raw) { fluxTransformer = value }
        if let raw = UserDefaults.standard.string(forKey: "fluxEncoder"),
           let value = Flux2FacadeEngine.FluxEncoderPrecision(rawValue: raw) { fluxEncoder = value }
        if let raw = UserDefaults.standard.string(forKey: "fluxDecoder"),
           let value = Flux2FacadeEngine.FluxDecoderPrecision(rawValue: raw) { fluxDecoder = value }
        Task { [libraryStore] in
            let restored = await libraryStore.load()
            await MainActor.run {
                // Merge, don't replace: a generation or delete that completed before this async restore
                // resolved must not be clobbered. Keep the in-session entries and add the disk records
                // not already present, newest first.
                let existing = Set(self.history.map(\.id))
                self.history = (self.history + restored.filter { !existing.contains($0.id) })
                    .sorted { $0.date > $1.date }
            }
        }
        #if os(iOS)
        // A paused generation pins the transformer + GPU graph (multiple GB) resident with no way for
        // iOS to reclaim it; under memory pressure or in the background that's a jetsam kill (looks like
        // a crash). Cancel a paused run on either signal — the user can re-generate.
        Task { [weak self] in
            for await _ in NotificationCenter.default.notifications(named: UIApplication.didReceiveMemoryWarningNotification) {
                await MainActor.run { self?.cancelPausedRunUnderMemoryPressure() }
            }
        }
        Task { [weak self] in
            for await _ in NotificationCenter.default.notifications(named: UIApplication.didEnterBackgroundNotification) {
                await MainActor.run { self?.cancelPausedRunUnderMemoryPressure() }
            }
        }
        #endif
    }

    #if os(iOS)
    /// Cancel a PAUSED generation under memory pressure / backgrounding so a multi-GB paused run isn't
    /// jetsammed. An active (un-paused) run is left to finish.
    private func cancelPausedRunUnderMemoryPressure() {
        if isGenerationPaused { cancelOperation() }
    }
    #endif

    var selected: DiffusionModel { models.first { $0.id == selectedID } ?? models[0] }

    /// Where downloaded models live (shown in Settings).
    var storageLocation: String { downloader.downloadBase.appending(component: "models").path }

    /// In-app download applies to Z-Image; FLUX manages its own weights inside `load`.
    var managesOwnDownload: Bool { selected.family != .zImage }

    var isDownloaded: Bool { isDownloaded(selected) }

    var isBusy: Bool {
        switch phase { case .downloading, .loading, .generating, .pausing, .paused, .cooling, .cancelling: return true; default: break }
        if fluxComponentDownloadID != nil { return true }   // a per-component install is in flight
        return false
    }
    var isFailed: Bool { if case .failed = phase { return true } else { return false } }

    var statusText: String {
        switch phase {
        case .idle: return managesOwnDownload ? "Ready to load" : (isDownloaded ? "Ready" : "Not downloaded")
        case .downloading(let f): return "Downloading… \(Int(f * 100))%"
        case .loading(let f): return "Loading into memory… \(Int(f * 100))%"
        case .generating(let s, let t): return "Generating… step \(s)/\(t)"
        case .pausing(let s, let t): return "Pausing after step \(s)/\(t)…"
        case .paused(let s, let t): return "Paused at step \(s)/\(t)"
        case .cooling(let s, let t): return "Cooling to protect your phone… (step \(s)/\(t))"
        case .cancelling: return "Cancelling…"
        case .cancelled: return "Cancelled"
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
        let operationID = UUID()
        activeOperationID = operationID
        defer { activeOperationID = nil }
        do {
            try await downloadSelected(operationID: operationID)
            componentsRevision += 1   // model-level download changed the on-disk component set
        } catch is CancellationError {
            phase = .cancelled
            componentsRevision += 1
        } catch {
            phase = .failed(String(describing: error))
        }
    }

    /// Reset Steps + Size to the selected model's native values, clamped per device (a phone defaults
    /// to 512 for memory). Called on launch and whenever the model changes, so the controls always
    /// reflect what the current model is calibrated for rather than a fixed global set.
    func applyModelDefaults() {
        steps = selected.defaultStepCount
        size = device.isPhone ? min(512, selected.nativeSize) : selected.nativeSize
    }

    /// Dismiss the current result and return the canvas to its empty state (the "×" on a finished
    /// generation). The image stays in the Library; this only clears the live canvas.
    func clearResult() {
        image = nil
        switch phase { case .done, .failed, .cancelled: phase = .idle; default: break }
    }

    /// Show a transient confirmation banner that auto-dismisses.
    func showToast(_ message: String) {
        toast = message
        toastTask?.cancel()
        toastTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 2_200_000_000)
            if !Task.isCancelled { toast = nil }
        }
    }

    func startGenerate() {
        // Guard against BOTH a running generation and a running install — and set inFlight + the task
        // handle SYNCHRONOUSLY (on the main actor) before the Task body runs, so a rapid install+generate
        // can't both pass their guards in the window before the async body sets inFlight (two pipelines).
        guard generationTask == nil, operationTask == nil, !inFlight,
              !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let operationID = UUID()
        let control = GenerationControl()
        inFlight = true
        generationControl = control
        activeOperationID = operationID
        generationTask = Task { @MainActor in
            defer {
                self.inFlight = false
                self.generationTask = nil
                self.generationControl = nil
                self.activeOperationID = nil
            }
            await self.runGenerate(control: control, operationID: operationID)
        }
    }

    func cancelOperation() {
        generationControl?.cancel()
        generationTask?.cancel()
        operationTask?.cancel()
        // Only show "Cancelling…" when a cancellable task actually exists; the handle-less inFlight
        // paths (model delete) can't be cancelled and would otherwise wedge on this label.
        if (generationTask != nil || operationTask != nil), isBusy { phase = .cancelling }
    }

    func pauseGeneration() {
        guard let control = generationControl else { return }
        control.pause()
        // Transition to .paused optimistically. The engine blocks at the NEXT step boundary, so no
        // further progress callback arrives to drive a .pausing→.paused transition — leaving it on
        // .pausing would stick "Pausing…" forever. (The in-flight step may still be finishing.)
        switch phase {
        case .generating(let step, let total), .pausing(let step, let total), .paused(let step, let total):
            phase = .paused(step, total)
        default:
            break
        }
    }

    func resumeGeneration() {
        guard let control = generationControl else { return }
        control.resume()
        switch phase {
        case .paused(let step, let total), .pausing(let step, let total):
            phase = .generating(step, total)
        default:
            break
        }
    }

    var canPauseGeneration: Bool {
        // Includes .cooling so the cancel/control cluster stays visible while auto-paused for heat —
        // the user can still cancel (the governor's cooling wait is cancellation-aware).
        switch phase { case .generating, .pausing, .paused, .cooling: return true; default: return false }
    }

    var isGenerationPaused: Bool {
        switch phase { case .pausing, .paused: return true; default: return false }
    }

    /// Backward-compatible async entry for previews/tests; the UI should call `startGenerate()` so
    /// the AppModel owns the cancellable task.
    func generate() async {
        startGenerate()
    }

    func startInstallRecipe(_ model: DiffusionModel) {
        guard operationTask == nil, generationTask == nil, !inFlight else { return }
        let operationID = UUID()
        inFlight = true
        activeOperationID = operationID
        operationTask = Task { @MainActor in
            defer {
                self.inFlight = false
                self.operationTask = nil
                self.activeOperationID = nil
            }
            self.selectedID = model.id
            do {
                try await self.downloadSelected(operationID: operationID)
                self.componentsRevision += 1
            } catch is CancellationError {
                self.phase = .cancelled
                self.componentsRevision += 1
            } catch {
                self.phase = .failed(String(describing: error))
            }
        }
    }

    func startInstallComponent(_ id: String, model: DiffusionModel) {
        guard operationTask == nil, generationTask == nil, !inFlight else { return }
        let operationID = UUID()
        inFlight = true
        activeOperationID = operationID
        operationTask = Task { @MainActor in
            defer {
                self.inFlight = false
                self.operationTask = nil
                self.activeOperationID = nil
            }
            if model.family == .flux2 {
                await self.downloadFluxComponentUnlocked(id, operationID: operationID)
            } else if model.family == .zImage {
                self.selectedID = model.id
                do {
                    try await self.downloadSelected(operationID: operationID)
                    self.componentsRevision += 1
                } catch is CancellationError {
                    self.phase = .cancelled
                    self.componentsRevision += 1
                } catch {
                    self.phase = .failed(String(describing: error))
                }
            }
        }
    }

    /// Delete a model's downloaded weights to free disk space. If that model is currently loaded,
    /// it is unloaded first so memory is freed too.
    func delete(_ model: DiffusionModel) async {
        guard !inFlight else { return }
        inFlight = true; defer { inFlight = false }
        if loadedID == model.id, let current = engine {
            engine = nil; loadedID = nil; loadedRecipe = nil; loadedFluxStreaming = nil
            await current.unload()
        }
        switch model.family {
        case .zImage:
            let url = downloader.localURL(repoId: model.variants[0].source.huggingFaceRepo)
            try? FileManager.default.removeItem(at: url)
        case .flux2:
            try? Flux2FacadeEngine.deleteWeights()
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
                PrecisionAxis(id: "decoder", title: "Decoder",
                    options: Flux2FacadeEngine.FluxDecoderPrecision.allCases.map { d in
                        #if os(iOS)
                        // The standard VAE is gated to 512 on iPhone (memory); make that visible.
                        let note = d == .standard ? "\(d.note) · 512 only" : d.note
                        #else
                        let note = d.note
                        #endif
                        return PrecisionOption(id: d.rawValue, label: d.label, note: note)
                    },
                    selectedID: fluxDecoder.rawValue),
            ]
            return ModelRecipe(axes: axes, components: comps)
        default:
            return ModelRecipe()
        }
    }

    /// Apply a precision-axis choice (FLUX only today).
    func setPrecision(axisID: String, optionID: String) {
        if axisID == "transformer", let v = Flux2FacadeEngine.FluxTransformerPrecision(rawValue: optionID) { fluxTransformer = v }
        if axisID == "encoder", let v = Flux2FacadeEngine.FluxEncoderPrecision(rawValue: optionID) { fluxEncoder = v }
        if axisID == "decoder", let v = Flux2FacadeEngine.FluxDecoderPrecision(rawValue: optionID) {
            fluxDecoder = v
            #if os(iOS)
            // Picking the standard VAE on a phone caps render size to 512 (its wider decoder channels
            // would balloon at 1024 on top of the resident 4-bit transformer).
            if v == .standard, size > 512 { size = 512 }
            #endif
        }
    }

    /// 0...1 progress for a component currently installing (nil if it isn't).
    func componentProgress(_ id: String, model: DiffusionModel) -> Double? {
        if model.family == .flux2, fluxComponentDownloadID == id { return fluxComponentFraction }
        if model.family == .zImage, id == "zimage", selectedID == model.id,
           case .downloading(let f) = phase { return f }
        return nil
    }

    /// A friendly error for a component whose last install failed (nil otherwise).
    func componentErrorMessage(_ id: String) -> String? {
        if fluxComponentError?.id == id { return fluxComponentError?.message }
        return nil
    }

    /// Install one recipe component.
    func installComponent(_ id: String, model: DiffusionModel) async {
        startInstallComponent(id, model: model)
    }

    /// Remove one recipe component.
    func removeComponent(_ id: String, model: DiffusionModel) async {
        if model.family == .flux2 { await deleteFluxComponent(id); return }
        if model.family == .zImage { await delete(model) }
    }

    /// Download all missing active components for a model (the model-level "Download" action).
    func installRecipe(_ model: DiffusionModel) async {
        startInstallRecipe(model)
    }

    /// Download the selected model's weights. No reentrancy guard — callers hold `inFlight`.
    private func downloadSelected(operationID: UUID) async throws {
        let model = selected
        #if os(iOS)
        // Keep the screen awake during the multi-GB download so a foreground URLSession isn't
        // killed by auto-lock mid-transfer.
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = false }
        #endif
        phase = .downloading(0)
        switch model.family {
        case .zImage:
            let repo = model.variants[0].source.huggingFaceRepo
            _ = try await downloader.download(repoId: repo) { fraction in
                Task { @MainActor in
                    if self.activeOperationID == operationID, case .downloading = self.phase {
                        self.phase = .downloading(fraction)
                    }
                }
            }
        case .flux2:
            try await Flux2FacadeEngine.download(transformer: fluxTransformer, encoder: fluxEncoder, decoder: fluxDecoder) { fraction in
                Task { @MainActor in
                    if self.activeOperationID == operationID, case .downloading = self.phase {
                        self.phase = .downloading(fraction)
                    }
                }
            }
        default:
            break
        }
        try Task.checkCancellation()
        phase = .idle
    }

    private func runGenerate(control: GenerationControl, operationID: UUID) async {
        let model = selected
        do {
            // Ensure the selected model's active recipe is on disk before loading (all families).
            if !isDownloaded(model) {
                try await downloadSelected(operationID: operationID)
                guard isDownloaded(model) else { return }
            }
            try control.checkpoint()

            // Reload when the model changed OR (FLUX) the chosen precision recipe changed.
            var needsReload = engine == nil || loadedID != model.id
            if model.family == .flux2 {
                if loadedRecipe != fluxRecipeLabel { needsReload = true }
                // 512 (resident facade) and 1024 (streaming engine) are different engines — rebuild
                // when the requested size crosses the boundary.
                if loadedFluxStreaming != fluxUsesStreaming { needsReload = true }
            }
            if needsReload {
                // Unload the previous engine BEFORE loading the new one, so two large weight sets
                // are never resident at once (a model switch would otherwise peak ~10+ GB).
                if let previous = engine {
                    engine = nil; loadedID = nil; loadedRecipe = nil; loadedFluxStreaming = nil
                    await previous.unload()
                }
                let built = try makeEngine(for: model)
                // The streaming Z-Image engine needs the real per-component source (it reads weights
                // through it); the resident facades ignore `source` and self-resolve their weights,
                // so they get a cheap empty placeholder.
                let source: WeightSource
                if model.family == .zImage && zImageUsesStreaming {
                    source = try zImageSource(for: model, streaming: true)
                } else if model.family == .flux2 && fluxUsesStreaming {
                    // The streaming FLUX engine reads the transformer through this source; the encoder
                    // and VAE load resident from their own caches.
                    source = try Flux2ComponentSource.openKlein4BStreaming()
                } else {
                    source = SafetensorsWeightSource(tensors: [:])
                }
                phase = .loading(0)
                try await built.load(model, variant: model.variants[0], source: source) { fraction in
                    Task { @MainActor in
                        if self.activeOperationID == operationID, case .loading = self.phase {
                            self.phase = .loading(fraction)
                        }
                    }
                }
                engine = built
                loadedID = model.id
                loadedRecipe = (model.family == .flux2) ? fluxRecipeLabel : nil
                loadedFluxStreaming = (model.family == .flux2) ? fluxUsesStreaming : nil
            }
            // Ensure the loaded engine is actually the selected model (a failed switch leaves none).
            guard loadedID == model.id, engine != nil else { phase = .failed("Model not loaded"); return }
            try control.checkpoint()

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
                                            size: ImageSize(width: size, height: size),
                                            control: control)
            let genStart = Date()
            let cgImage: CGImage
            do {
                guard let currentEngine = engine else { phase = .failed("Model not loaded"); return }
                cgImage = try await currentEngine.generate(request) { progress in
                    Task { @MainActor in
                        // Only advance while still generating, so a late callback can't revive a finished run.
                        guard self.activeOperationID == operationID else { return }
                        switch progress {
                        case .denoising(let step, let total, _):
                            // A denoise step after a cooling pause means the device cooled and the run
                            // auto-resumed — leave .cooling and reflect live progress again.
                            switch self.phase {
                            case .generating, .pausing, .paused, .cooling:
                                self.phase = control.isPaused ? .paused(step, total) : .generating(step, total)
                            default:
                                break
                            }
                        case .cooling:
                            // Carry the last known step into the cooling phase so progress context survives.
                            switch self.phase {
                            case .generating(let s, let t), .pausing(let s, let t),
                                 .paused(let s, let t), .cooling(let s, let t):
                                self.phase = .cooling(s, t)
                            default:
                                break
                            }
                        default:
                            break
                        }
                    }
                }
            }
            try control.checkpoint()
            await unloadOneShotStreamingEngineIfNeeded(for: model)
            // Defensive: only the active operation may write the terminal result, so a stale run that
            // finished after a newer one started can't stomp its phase / insert its image.
            guard activeOperationID == operationID else { return }
            let elapsed = Date().timeIntervalSince(genStart)
            lastGenerationSeconds = elapsed
            image = cgImage
            phase = .done
            let generation = Generation(image: cgImage, prompt: prompt, modelID: model.id,
                                        modelName: model.displayName, size: size, steps: steps, seed: seed,
                                        duration: elapsed, settings: generationSettings(for: model))
            history.insert(generation, at: 0)
            do {
                try await libraryStore.save(history)
                showToast("Saved to Library")
            } catch {
                showToast("Couldn’t save to Library")
            }
        } catch is CancellationError {
            await unloadOneShotStreamingEngineIfNeeded(for: model)
            guard activeOperationID == operationID else { return }
            phase = .cancelled
            showToast("Cancelled")
        } catch EngineError.pausedForHeat {
            // The device stayed too hot past the cooling window. This is recoverable, not a failure:
            // return to idle and invite a retry once it cools, rather than showing a scary error.
            await unloadOneShotStreamingEngineIfNeeded(for: model)
            guard activeOperationID == operationID else { return }
            phase = .idle
            showToast("Paused to let your phone cool — try again in a moment")
        } catch {
            await unloadOneShotStreamingEngineIfNeeded(for: model)
            guard activeOperationID == operationID else { return }
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
            // The facade is phone-aware: Mac → resident, iPhone → two-phase with the pre-quantized
            // 4-bit checkpoint, gated against the device's memory budget.
            return Flux2FacadeEngine.capabilities(for: model, variant: variant, on: device)
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
            return Flux2FacadeEngine.isDownloaded(transformer: fluxTransformer, encoder: fluxEncoder, decoder: fluxDecoder)
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
            total += Flux2FacadeEngine.downloadedBytes()   // FLUX weights live in the engine's own cache
            return total
        }.value
    }

    /// The model-specific recipe rows to record with a generation (and show in the Library detail).
    /// Extensible per family: each contributes whatever settings are meaningful; a family with nothing
    /// extra just falls through to the common Model/Size/Steps/Seed rows. New families add a case here.
    func generationSettings(for model: DiffusionModel) -> [GenerationSetting] {
        switch model.family {
        case .flux2:
            return [
                GenerationSetting(label: "Transformer", value: fluxTransformer.label,
                                  axisID: "transformer", optionID: fluxTransformer.rawValue),
                GenerationSetting(label: "Text encoder", value: fluxEncoder.label,
                                  axisID: "encoder", optionID: fluxEncoder.rawValue),
                GenerationSetting(label: "Decoder", value: fluxDecoder.label,
                                  axisID: "decoder", optionID: fluxDecoder.rawValue),
            ]
        case .zImage:
            // Z-Image ships a single fixed-precision variant; surface it so the readout is complete.
            return model.variants.first.map { [GenerationSetting(label: "Precision", value: $0.precision.label)] } ?? []
        default:
            return []   // future families (e.g. qwenImage) add their own rows here
        }
    }

    /// Apply a past generation's settings and jump to Create — including the model-specific recipe
    /// (precision / decoder) so the next run reproduces this image, not just the prompt/size/seed.
    func reuse(_ g: Generation) {
        prompt = g.prompt; size = g.size; steps = g.steps; seedText = String(g.seed)
        if models.contains(where: { $0.id == g.modelID }) { selectedID = g.modelID }
        for s in g.settings {
            if let axisID = s.axisID, let optionID = s.optionID { setPrecision(axisID: axisID, optionID: optionID) }
        }
        tab = .create
    }

    /// Delete one saved Library record and its image file.
    func deleteGeneration(_ generation: Generation) {
        history.removeAll { $0.id == generation.id }
        let remaining = history
        Task { [libraryStore] in
            do {
                try await libraryStore.delete(generation, remaining: remaining)
                await MainActor.run { self.showToast("Deleted") }
            } catch {
                await MainActor.run { self.showToast("Couldn’t delete") }
            }
        }
    }

    /// Z-Image runs the block-streaming `MLXDiffusionEngine` on iPhone (partial-load, ~3 GB peak)
    /// and stays on the resident `ZImageFacadeEngine` on Mac (it already produces correct images).
    private var zImageUsesStreaming: Bool { device.isPhone }

    /// FLUX: 1024 streams the transformer block-by-block (the only way it fits an iPhone); 512 stays on
    /// the fast, validated resident facade. macOS always runs the resident facade.
    private var fluxUsesStreaming: Bool { device.isPhone && size > 512 }

    private func unloadOneShotStreamingEngineIfNeeded(for model: DiffusionModel) async {
        guard model.family == .zImage, zImageUsesStreaming, loadedID == model.id,
              let current = engine else { return }
        engine = nil
        loadedID = nil
        loadedRecipe = nil
        loadedFluxStreaming = nil
        await current.unload()
    }

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
            if fluxUsesStreaming {
                // iPhone 1024: the generic MLX engine drives Flux2Architecture, streaming each of the
                // 25 transformer blocks per step from the component source (encoder + VAE load resident
                // from their own caches). 512 falls through to the resident facade below.
                return MLXDiffusionEngine(architecture: Flux2Architecture(vaeVariant: fluxDecoder.vae),
                                          device: device)
            }
            return Flux2FacadeEngine(transformer: fluxTransformer, encoder: fluxEncoder, decoder: fluxDecoder)
        default:
            throw AppError.unsupportedOnPlatform("\(model.displayName) is not supported yet.")
        }
    }

    // MARK: - Image export

    /// PNG bytes for a `CGImage`, used by the macOS save panel / `.fileExporter`. Cross-platform.
    func pngData(_ cg: CGImage) -> Data? {
        ImageCodec.pngData(cg)
    }

    #if os(iOS)
    /// Save a generated image to the user's Photo library (add-only authorization). Denial is
    /// non-fatal — the image stays in the in-app Library. iOS only; macOS uses a save panel in-view.
    func exportImage(_ cg: CGImage) {
        let image = UIImage(cgImage: cg)
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                Task { @MainActor in self.showToast("Photos access denied") }
                return
            }
            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            } completionHandler: { success, _ in
                Task { @MainActor in self.showToast(success ? "Saved to Photos" : "Couldn’t save") }
            }
        }
    }
    #endif
}
