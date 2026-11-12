import SwiftUI
@preconcurrency import StableDiffusion
import CoreML

@MainActor
class DiffusionViewModel: ObservableObject {
    @Published var prompt: String = "A cinematic portrait of a robot in a tailored suit"
    @Published var generatedImage: UIImage? = nil
    @Published var isGenerating: Bool = false
    @Published var generationTime: Double = 0.0
    @Published var elapsedGenerationTime: Double = 0.0
    @Published var isResourcesValid: Bool = false
    @Published var statusMessage: String = "Checking SD3 resources."
    @Published var errorMessage: String? = nil
    @Published var memoryFootprintMB: Double = 0
    @Published private(set) var selectedModel: DiffusionModelKind = .sd3MediumTwoStep

    @Published private(set) var resolution: SD3Resolution = .default
    /// ReSeed alpha for the 4-step model (0 = off, 1 = paper default). Same weights either way,
    /// so changing it never reloads the pipeline.
    @Published var reseedScale: Float = 0

    /// Compute-units fallback chain. Keep this ANE-only while validating MMDiT
    /// acceleration: CPU/GPU fallback can hide an ANE load failure and make
    /// generation look "working" while being far too slow for the target UX.
    private let computeProfiles: [SD3PipelineLoader.ComputeUnitsProfile] =
        [.aneFirst]

    private var pipeline: (any StableDiffusionPipelineProtocol)?
    private var kleinPipeline: Flux2KleinPipeline?
    private let cancelFlag = CancelFlag()
    @Published private(set) var isCancelling: Bool = false

    /// Ask the running generation to stop. SD3 stops after the current denoising step; FLUX.2 Klein
    /// stops before the next Core ML stage.
    func cancelGeneration() {
        guard isGenerating, !isCancelling else { return }
        cancelFlag.set(true)
        isCancelling = true
        statusMessage = "Cancelling…"
    }

    fileprivate func finishCancelled() {
        isGenerating = false
        isCancelling = false
        cancelFlag.set(false)
        stopGenerationTimer()
        generationTime = 0
        elapsedGenerationTime = 0
        statusMessage = "Generation cancelled."
        errorMessage = nil
        if pendingMemoryWarningUnload { unloadPipeline(reason: "deferred-memoryWarning") }
    }

    func selectFamily(_ family: ModelFamily) {
        selectModel(.kind(family: family, steps: selectedModel.stepCount))
    }

    func selectSteps(_ steps: Int) {
        selectModel(.kind(family: selectedModel.family, steps: steps))
    }
    private var resourceURL: URL?
    private var activeProfile: SD3PipelineLoader.ComputeUnitsProfile?
    private var loadedModel: DiffusionModelKind?
    private var loadedSD3Resolution: SD3Resolution?
    private var timerTask: Task<Void, Never>?
    private var pendingMemoryWarningUnload = false

    private struct GeneratedImageSnapshot {
        let image: UIImage
        let generationTime: Double
        let memoryFootprintMB: Double
    }

    private struct SnapshotKey: Hashable {
        let model: DiffusionModelKind
        let resolution: SD3Resolution
    }

    private var generatedImagesByModel: [SnapshotKey: GeneratedImageSnapshot] = [:]

    private var snapshotKey: SnapshotKey {
        SnapshotKey(model: selectedModel, resolution: resolution)
    }

    private func logElapsed(_ label: String, since start: CFAbsoluteTime) {
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        print(String(format: "[SD3] timing: %@ %.3fs", label, elapsed))
    }

    /// Cheap check on app launch: does the bundle actually contain the models?
    /// Does NOT instantiate any Core ML pipeline (those load on first Generate).
    func validateResources() {
        validateSD3Resources()
        MemoryProbe.log("at app launch")
    }

    func selectModel(_ model: DiffusionModelKind) {
        guard model != selectedModel else { return }
        guard !isGenerating else { return }

        cacheCurrentGeneratedImage()
        unloadPipeline(reason: "model-switch")
        selectedModel = model
        if !model.supportedResolutions.contains(resolution) {
            resolution = model.supportedResolutions[0]
        }
        restoreGeneratedImage()
        validateSD3Resources()
    }

    func selectResolution(_ newResolution: SD3Resolution) {
        guard newResolution != resolution else { return }
        guard !isGenerating else { return }
        guard selectedModel.supportedResolutions.contains(newResolution) else { return }

        cacheCurrentGeneratedImage()
        unloadPipeline(reason: "resolution-switch")
        resolution = newResolution
        restoreGeneratedImage()
        validateSD3Resources()
    }

    var modelDisplayName: String {
        selectedModel.displayName
    }

    var modelShortName: String {
        selectedModel.shortName
    }

    var activeResolutionLabel: String {
        resolution.label
    }

    private func validateSD3Resources() {
        let folderName = selectedModel.resourceFolderName(for: resolution)
        let resolved = selectedModel.isKlein
            ? SD3PipelineLoader.resolveKleinResourceURL(folderName: folderName)
            : SD3PipelineLoader.resolveResourceURL(folderName: folderName)
        guard let url = resolved else {
            errorMessage = "\(selectedModel.shortName) \(resolution.label) resources not found. Open Settings to download \(folderName)."
            statusMessage = "\(selectedModel.displayName) \(resolution.label) resources not found. Download models in Settings."
            resourceURL = nil
            isResourcesValid = false
            return
        }

        let missing = SD3PipelineLoader.missingResources(at: url, for: selectedModel)
        resourceURL = url
        isResourcesValid = missing.isEmpty
        if missing.isEmpty {
            statusMessage = "\(selectedModel.displayName) \(resolution.label) models found at \(url.lastPathComponent). First Generate will load them (~10–30s)."
            errorMessage = nil
        } else {
            statusMessage = "\(selectedModel.displayName) \(resolution.label) resources incomplete: \(missing.joined(separator: ", "))"
            let shown = missing.prefix(4).joined(separator: ", ") + (missing.count > 4 ? " … (\(missing.count) files)" : "")
            errorMessage = "Missing \(selectedModel.selectorLabel) resources: \(shown)"
        }
        print("[\(selectedModel.shortName)] \(resolution.label) resources at \(url.path)")
    }

    private func cacheCurrentGeneratedImage() {
        guard let generatedImage else { return }
        generatedImagesByModel[snapshotKey] = GeneratedImageSnapshot(
            image: generatedImage,
            generationTime: generationTime,
            memoryFootprintMB: memoryFootprintMB
        )
    }

    private func restoreGeneratedImage() {
        if let snapshot = generatedImagesByModel[snapshotKey] {
            generatedImage = snapshot.image
            generationTime = snapshot.generationTime
            elapsedGenerationTime = snapshot.generationTime
            memoryFootprintMB = snapshot.memoryFootprintMB
        } else {
            generatedImage = nil
            generationTime = 0
            elapsedGenerationTime = 0
            memoryFootprintMB = 0
        }
        errorMessage = nil
    }

    /// Tear down the entire pipeline. ARC drops every ManagedMLModel which
    /// frees ANE/GPU resources. Use on memory warnings or before retrying with
    /// a different compute-units profile.
    func unloadPipeline(reason: String) {
        if kleinPipeline != nil {
            // Klein keeps no Core ML model resident between runs; dropping it frees the tokenizer.
            kleinPipeline = nil
            loadedModel = nil
            loadedSD3Resolution = nil
            print("[Klein] Pipeline unloaded (\(reason))")
        }
        if pipeline != nil {
            pipeline?.unloadResources()
            pipeline = nil
            activeProfile = nil
            loadedModel = nil
            loadedSD3Resolution = nil
            print("[SD3] Pipeline unloaded (\(reason))")
            MemoryProbe.log("after unload[\(reason)]")
        }
        pendingMemoryWarningUnload = false
    }

    func handleMemoryWarning() {
        if isGenerating {
            pendingMemoryWarningUnload = true
            print("[SD3] memory warning during generation; deferring pipeline unload")
            MemoryProbe.log("deferred unload[memoryWarning]")
        } else {
            unloadPipeline(reason: "memoryWarning")
        }
    }

    private func startGenerationTimer() -> CFAbsoluteTime {
        let startTime = CFAbsoluteTimeGetCurrent()
        generationTime = 0
        elapsedGenerationTime = 0
        timerTask?.cancel()
        timerTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                self?.elapsedGenerationTime = CFAbsoluteTimeGetCurrent() - startTime
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
        return startTime
    }

    private func stopGenerationTimer(finalTime: Double? = nil) {
        timerTask?.cancel()
        timerTask = nil
        if let finalTime {
            generationTime = finalTime
            elapsedGenerationTime = finalTime
        } else if elapsedGenerationTime > 0 {
            generationTime = elapsedGenerationTime
        }
    }

    fileprivate func finishKlein(image: UIImage, finalTime: Double) {
        stopGenerationTimer(finalTime: finalTime)
        memoryFootprintMB = MemoryProbe.residentMB()
        generatedImage = image
        generatedImagesByModel[snapshotKey] = GeneratedImageSnapshot(
            image: image, generationTime: finalTime, memoryFootprintMB: memoryFootprintMB
        )
        statusMessage = "FLUX.2 Klein ready."
        isGenerating = false
        if pendingMemoryWarningUnload { unloadPipeline(reason: "deferred-memoryWarning") }
    }

    fileprivate func failKlein(_ error: Error) {
        isGenerating = false
        unloadPipeline(reason: "generate-error")
        stopGenerationTimer()
        errorMessage = "Generation failed: \(error.localizedDescription)"
    }

    func generate() async {
        guard !prompt.isEmpty, let resourceURL, isResourcesValid else { return }

        isGenerating = true
        isCancelling = false
        cancelFlag.set(false)
        errorMessage = nil
        let startTime = startGenerationTimer()

        if selectedModel.isKlein {
            await generateKlein(resourceURL: resourceURL, startTime: startTime)
            return
        }

        // First-time pipeline construction (with fallback compute-units chain)
        if pipeline == nil
            || loadedModel != selectedModel
            || loadedSD3Resolution != resolution
        {
            statusMessage = "Loading models (compiling Core ML execution plan)…"
            MemoryProbe.log("before pipeline build")
            let pipelineBuildStart = CFAbsoluteTimeGetCurrent()
            do {
                let (pipe, profile) = try await Task.detached(priority: .userInitiated) {
                    [computeProfiles] in
                    try SD3PipelineLoader.createPipelineWithFallback(
                        at: resourceURL,
                        profiles: computeProfiles
                    )
                }.value
                pipeline = pipe
                activeProfile = profile
                loadedModel = selectedModel
                loadedSD3Resolution = resolution
                statusMessage = "Pipeline ready (compute=\(profile.rawValue))"
                MemoryProbe.log("after pipeline build")
                logElapsed("pipeline build", since: pipelineBuildStart)
            } catch {
                isGenerating = false
                stopGenerationTimer()
                errorMessage = "Failed to load: \(error.localizedDescription)"
                print("[SD3] Pipeline construction failed: \(error)")
                return
            }
        }

        guard let pipeline else {
            isGenerating = false
            stopGenerationTimer()
            return
        }

        // Yield one frame so the spinner appears before MMDiT pegs the cores.
        try? await Task.sleep(nanoseconds: 100_000_000)

        var config = PipelineConfiguration(prompt: prompt)
        // Distilled SD3-Medium schedule:
        //   t     = linspace(1.0, 0.0, stepCount + 1)
        //   sigma = 3*t / (1 + 2*t)
        //   update: x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v
        // The distilled weights are not CFG-aware; pipeline auto-detects
        // batch=1 MMDiT and skips uncond branch.
        config.stepCount = selectedModel.stepCount
        config.guidanceScale = selectedModel.guidanceScale
        config.schedulerTimestepShift = selectedModel.timestepShift
        config.reseedScale = selectedModel.reseedScales.isEmpty ? 0 : reseedScale
        config.seed = UInt32.random(in: UInt32.min...UInt32.max)
        print("[SD3] seed: \(config.seed)")
        config.encoderScaleFactor = 1.5305
        config.decoderScaleFactor = 1.5305
        config.decoderShiftFactor = 0.0609
        config.originalSize = Float32(resolution.rawValue)
        config.targetSize = Float32(resolution.rawValue)

        do {
            MemoryProbe.log("before generateImages")
            let inferenceStart = CFAbsoluteTimeGetCurrent()
            let cgImage = try await Task.detached(priority: .userInitiated) { [cancelFlag] in
                try autoreleasepool { () -> CGImage? in
                    let result = try pipeline.generateImages(
                        configuration: config,
                        progressHandler: { progress in
                            // Per-step memory probe so we can pinpoint OOM
                            // location: conditioning vs. step-N MMDiT vs. VAE decode.
                            MemoryProbe.log("step \(progress.step + 1)/\(progress.stepCount)")
                            return !cancelFlag.isSet
                        }
                    )
                    MemoryProbe.log("after MMDiT loop / before VAE decode")
                    return result.first.flatMap { $0 }
                }
            }.value
            MemoryProbe.log("after generateImages")
            logElapsed("generateImages", since: inferenceStart)
            if cancelFlag.isSet {
                finishCancelled()
                return
            }

            let finalTime = CFAbsoluteTimeGetCurrent() - startTime
            stopGenerationTimer(finalTime: finalTime)
            memoryFootprintMB = MemoryProbe.residentMB()
            if let cgImage {
                let image = UIImage(cgImage: cgImage)
                generatedImage = image
                generatedImagesByModel[snapshotKey] = GeneratedImageSnapshot(
                    image: image,
                    generationTime: finalTime,
                    memoryFootprintMB: memoryFootprintMB
                )
            }
            isGenerating = false
            if pendingMemoryWarningUnload {
                unloadPipeline(reason: "deferred-memoryWarning")
            }
        } catch {
            print("[SD3] Generation failed: \(error)")
            isGenerating = false
            unloadPipeline(reason: "generate-error")
            stopGenerationTimer()
            errorMessage = "Generation failed: \(error.localizedDescription)"
        }
    }
}

extension DiffusionViewModel {
    /// FLUX.2 Klein path: every stage (text, denoiser, VAE) is loaded, run and released in turn.
    fileprivate func generateKlein(resourceURL: URL, startTime: CFAbsoluteTime) async {
        do {
            if kleinPipeline == nil {
                statusMessage = "Loading FLUX.2 Klein tokenizer…"
                let built = try await Task.detached(priority: .userInitiated) { () -> Flux2KleinPipeline in
                    let config = MLModelConfiguration()
                    config.computeUnits = .cpuAndNeuralEngine
                    return try Flux2KleinPipeline(resourcesAt: resourceURL, configuration: config)
                }.value
                kleinPipeline = built
                loadedModel = selectedModel
                loadedSD3Resolution = resolution
            }
            guard let klein = kleinPipeline else { return }
            try? await Task.sleep(nanoseconds: 100_000_000)
            let seed = UInt32.random(in: UInt32.min...UInt32.max)
            let steps = selectedModel.stepCount
            let prompt = self.prompt
            print("[Klein] seed: \(seed)")
            MemoryProbe.log("before Klein generate")
            let cancelFlag = self.cancelFlag
            let (cgImage, timings) = try await Task.detached(priority: .userInitiated) { [weak self] in
                let image = try klein.generateImage(prompt: prompt, seed: seed, steps: steps,
                                                    isCancelled: { cancelFlag.isSet }) { phase in
                    let text: String
                    switch phase {
                    case .encodingText: text = "Encoding prompt (Qwen3 on ANE)…"
                    case .denoising(let step, let total): text = "Denoising step \(step)/\(total)"
                    case .decoding: text = "Decoding image (tiled VAE)…"
                    }
                    MemoryProbe.log("Klein \(text)")
                    Task { @MainActor [weak self] in
                        if self?.isCancelling == false { self?.statusMessage = text }
                    }
                    return !cancelFlag.isSet
                }
                return (image, klein.lastTimings)
            }.value
            print(String(format: "[Klein] timing: text %.1fs denoise %.1fs decode %.1fs (model load %.1fs)",
                         timings.text, timings.denoise, timings.decode, timings.load))
            MemoryProbe.log("after Klein generate")
            let finalTime = CFAbsoluteTimeGetCurrent() - startTime
            finishKlein(image: UIImage(cgImage: cgImage), finalTime: finalTime)
        } catch Flux2KleinPipeline.KleinError.cancelled {
            print("[Klein] Generation cancelled")
            finishCancelled()
        } catch {
            print("[Klein] Generation failed: \(error)")
            failKlein(error)
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = DiffusionViewModel()
    @StateObject private var resourceManager = ModelResourceManager()
    @Environment(\.colorScheme) private var colorScheme
    @FocusState private var isPromptFocused: Bool
    @State private var isSettingsPresented = false

    private var themeTextColor: Color {
        colorScheme == .dark ? .white : .black
    }

    private var secondaryThemeTextColor: Color {
        themeTextColor.opacity(0.65)
    }

    private var canGenerate: Bool {
        !viewModel.isGenerating
            && !viewModel.prompt.isEmpty
            && viewModel.isResourcesValid
    }

    var body: some View {
        VStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Spacer()

                    Button {
                        isSettingsPresented = true
                    } label: {
                        Image(systemName: "gearshape")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(themeTextColor)
                            .frame(width: 36, height: 36)
                            .background(Color(UIColor.secondarySystemBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    }
                    .disabled(viewModel.isGenerating)
                }

                Picker(
                    "Model",
                    selection: Binding(
                        get: { viewModel.selectedModel.family },
                        set: { viewModel.selectFamily($0) }
                    )
                ) {
                    ForEach(ModelFamily.allCases) { family in
                        Text(family.label).tag(family)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(viewModel.isGenerating)

                HStack(alignment: .center, spacing: 10) {
                    Picker(
                        "Diffusion steps",
                        selection: Binding(
                            get: { viewModel.selectedModel.stepCount },
                            set: { viewModel.selectSteps($0) }
                        )
                    ) {
                        ForEach(DiffusionModelKind.stepChoices, id: \.self) { steps in
                            Text("\(steps) steps").tag(steps)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(viewModel.isGenerating)

                    Picker(
                        "Resolution",
                        selection: Binding(
                            get: { viewModel.resolution },
                            set: { viewModel.selectResolution($0) }
                        )
                    ) {
                        ForEach(SD3Resolution.allCases) { resolution in
                            Text(resolution.shortLabel).tag(resolution)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 120)
                    .disabled(viewModel.isGenerating)
                }

                if !viewModel.selectedModel.reseedScales.isEmpty {
                    Picker("ReSeed strength", selection: $viewModel.reseedScale) {
                        ForEach(viewModel.selectedModel.reseedScales, id: \.self) { scale in
                            Text(scale == 0 ? "ReSeed off" : String(format: "α %.1f", scale)).tag(scale)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(viewModel.isGenerating)
                }

                ZStack(alignment: .topLeading) {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color(UIColor.secondarySystemBackground))

                    if viewModel.prompt.isEmpty {
                        Text("Prompt")
                            .font(.system(size: 19))
                            .foregroundColor(themeTextColor.opacity(0.45))
                            .padding(.top, 7)
                            .padding(.leading, 12)
                    }

                    TextEditor(text: $viewModel.prompt)
                        .font(.system(size: 19))
                        .foregroundColor(themeTextColor)
                        .frame(height: 64)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 0)
                        .scrollContentBackground(.hidden)
                        .focused($isPromptFocused)
                }
                .frame(height: 64)
            }
            .padding(.horizontal, 16)
            .padding(.top, 10)

            ZStack {
                if let image = viewModel.generatedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .cornerRadius(16)
                } else {
                    Rectangle()
                        .fill(Color(UIColor.systemGray6))
                        .cornerRadius(16)

                    if let error = viewModel.errorMessage {
                        VStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundColor(.red)
                                .font(.system(size: 30))
                            Text(error)
                                .font(.system(size: 14))
                                .foregroundColor(.red)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                        }
                    } else if viewModel.isGenerating {
                        VStack(spacing: 12) {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle())
                                .scaleEffect(1.5)
                            Text(String(format: "Generating %.1fs", viewModel.elapsedGenerationTime))
                                .font(.system(size: 18, weight: .semibold))
                                .foregroundColor(themeTextColor)
                            Text(viewModel.statusMessage)
                                .font(.system(size: 12))
                                .foregroundColor(secondaryThemeTextColor)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                        }
                    } else if !viewModel.isResourcesValid {
                        VStack(spacing: 12) {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle())
                                .scaleEffect(1.5)
                            Text("Checking \(viewModel.modelShortName) Resources...")
                                .font(.system(size: 16, weight: .medium))
                                .foregroundColor(secondaryThemeTextColor)
                            Text(viewModel.statusMessage)
                                .font(.system(size: 11))
                                .foregroundColor(secondaryThemeTextColor)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                        }
                    } else {
                        VStack(spacing: 8) {
                            Text("Enter prompt to generate")
                                .foregroundColor(secondaryThemeTextColor)
                            Text(viewModel.statusMessage)
                                .font(.system(size: 11))
                                .foregroundColor(secondaryThemeTextColor)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                        }
                    }
                }
            }
            .aspectRatio(1, contentMode: .fit)
            .padding(.horizontal, 16)
            .contentShape(Rectangle())
            .onTapGesture {
                isPromptFocused = false
            }

            HStack(spacing: 10) {
                if viewModel.isGenerating {
                    Button(action: {
                        viewModel.cancelGeneration()
                    }) {
                        HStack(spacing: 8) {
                            Image(systemName: viewModel.isCancelling ? "hourglass" : "xmark.circle.fill")
                            Text(viewModel.isCancelling ? "Cancelling" : "Cancel")
                        }
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity, minHeight: 48)
                        .background(viewModel.isCancelling ? Color.gray : Color.red)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    }
                    .disabled(viewModel.isCancelling)
                } else {
                    Button(action: {
                        Task { await viewModel.generate() }
                    }) {
                        HStack(spacing: 8) {
                            Image(systemName: "sparkles")
                            Text("Generate")
                        }
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity, minHeight: 48)
                        .background(canGenerate ? Color.blue : Color.gray)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    }
                    .disabled(!canGenerate)
                }

                if let image = viewModel.generatedImage {
                    ShareLink(
                        item: Image(uiImage: image),
                        preview: SharePreview("Generated Image", image: Image(uiImage: image))
                    ) {
                        Image(systemName: "square.and.arrow.up")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(.blue)
                            .frame(width: 48, height: 48)
                            .background(Color(UIColor.secondarySystemBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    }
                } else {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.gray)
                        .frame(width: 48, height: 48)
                        .background(Color(UIColor.secondarySystemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
            }
            .padding(.horizontal, 16)

            HStack {
                if viewModel.isGenerating {
                    Text(String(format: "Generating %.1fs", viewModel.elapsedGenerationTime))
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(themeTextColor)
                } else if viewModel.generationTime > 0 {
                    Text(String(format: "%.1fs · %.0f MB",
                                viewModel.generationTime,
                                viewModel.memoryFootprintMB))
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(themeTextColor)
                } else {
                    Text("Idle")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(secondaryThemeTextColor)
                }

                Spacer()

                Text(viewModel.modelShortName)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(secondaryThemeTextColor)
            }
            .padding(.horizontal, 16)

            Spacer()
        }
        .onAppear {
            viewModel.validateResources()
        }
        .onReceive(NotificationCenter.default.publisher(
            for: UIApplication.didReceiveMemoryWarningNotification
        )) { _ in
            viewModel.handleMemoryWarning()
        }
        .onReceive(NotificationCenter.default.publisher(
            for: UIApplication.didEnterBackgroundNotification
        )) { _ in
            viewModel.unloadPipeline(reason: "backgrounded")
        }
        .sheet(isPresented: $isSettingsPresented) {
            ModelSettingsView(
                resourceManager: resourceManager,
                selectedModel: viewModel.selectedModel,
                resolution: viewModel.resolution,
                onResourcesChanged: {
                    viewModel.unloadPipeline(reason: "resources-updated")
                    viewModel.validateResources()
                }
            )
        }
    }
}

struct ModelSettingsView: View {
    @ObservedObject var resourceManager: ModelResourceManager
    let selectedModel: DiffusionModelKind
    let resolution: SD3Resolution
    let onResourcesChanged: () -> Void

    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var colorScheme

    private var themeTextColor: Color {
        colorScheme == .dark ? .white : .black
    }

    private var secondaryTextColor: Color {
        themeTextColor.opacity(0.65)
    }

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Model Resources")
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(themeTextColor)

                    Text("Download SD3 Medium and FLUX.2 Klein Core ML resources into this app. The files are stored locally and reused across launches.")
                        .font(.system(size: 14))
                        .foregroundColor(secondaryTextColor)
                }

                VStack(alignment: .leading, spacing: 12) {
                    ForEach(SD3Resolution.allCases) { res in
                        ForEach(DiffusionModelKind.resourceKinds) { model in
                            ResourceStatusRow(
                                title: "\(model.resourceLabel) · \(res.label)",
                                isReady: resourceManager.hasResources(for: model, resolution: res)
                            )
                        }
                    }
                }

                if resourceManager.isDownloading {
                    VStack(alignment: .leading, spacing: 8) {
                        ProgressView(value: resourceManager.progress)
                            .progressViewStyle(.linear)
                        Text(resourceManager.statusMessage)
                            .font(.system(size: 12))
                            .foregroundColor(secondaryTextColor)
                            .lineLimit(3)
                    }
                } else {
                    Text(resourceManager.statusMessage)
                        .font(.system(size: 12))
                        .foregroundColor(secondaryTextColor)
                        .lineLimit(3)
                }

                VStack(spacing: 10) {
                    Button {
                        Task {
                            await resourceManager.downloadSelected(selectedModel, resolution: resolution)
                            onResourcesChanged()
                        }
                    } label: {
                        Label("Download \(selectedModel.resourceLabel) · \(resolution.label)", systemImage: "arrow.down.circle")
                            .frame(maxWidth: .infinity, minHeight: 46)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(resourceManager.isDownloading)

                    Button {
                        Task {
                            await resourceManager.downloadAll()
                            onResourcesChanged()
                        }
                    } label: {
                        Label("Download SD3M 2-Step and 4-Step (512)", systemImage: "square.and.arrow.down")
                            .frame(maxWidth: .infinity, minHeight: 46)
                    }
                    .buttonStyle(.bordered)
                    .disabled(resourceManager.isDownloading)
                }

                Spacer()
            }
            .padding(20)
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        onResourcesChanged()
                        dismiss()
                    }
                }
            }
        }
    }
}

private struct ResourceStatusRow: View {
    let title: String
    let isReady: Bool

    var body: some View {
        HStack {
            Text(title)
                .font(.system(size: 16, weight: .semibold))

            Spacer()

            Label(isReady ? "Ready" : "Missing", systemImage: isReady ? "checkmark.circle.fill" : "arrow.down.circle")
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(isReady ? .green : .orange)
        }
        .padding(12)
        .background(Color(UIColor.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

#Preview {
    ContentView()
}
