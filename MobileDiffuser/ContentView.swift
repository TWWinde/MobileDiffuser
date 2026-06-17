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

    private let resolution: SD3Resolution = .default

    /// Compute-units fallback chain. Keep this ANE-only while validating MMDiT
    /// acceleration: CPU/GPU fallback can hide an ANE load failure and make
    /// generation look "working" while being far too slow for the target UX.
    private let computeProfiles: [SD3PipelineLoader.ComputeUnitsProfile] =
        [.aneFirst]

    private var pipeline: (any StableDiffusionPipelineProtocol)?
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

    private var generatedImagesByModel: [DiffusionModelKind: GeneratedImageSnapshot] = [:]

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
        restoreGeneratedImage(for: model)
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
        guard let url = SD3PipelineLoader.resolveResourceURL(
            folderName: folderName
        ) else {
            errorMessage = "\(selectedModel.shortName) \(resolution.label) resources not found. Open Settings to download \(folderName)."
            statusMessage = "\(selectedModel.displayName) \(resolution.label) resources not found. Download models in Settings."
            resourceURL = nil
            isResourcesValid = false
            return
        }

        let missing = SD3PipelineLoader.missingResources(at: url)
        resourceURL = url
        isResourcesValid = missing.isEmpty
        if missing.isEmpty {
            statusMessage = "\(selectedModel.displayName) \(resolution.label) models found at \(url.lastPathComponent). First Generate will load them (~10–30s)."
            errorMessage = nil
        } else {
            statusMessage = "\(selectedModel.displayName) \(resolution.label) resources incomplete: \(missing.joined(separator: ", "))"
            errorMessage = "Missing \(selectedModel.selectorLabel) resources: \(missing.joined(separator: ", "))"
        }
        print("[\(selectedModel.shortName)] \(resolution.label) resources at \(url.path)")
    }

    private func cacheCurrentGeneratedImage() {
        guard let generatedImage else { return }
        generatedImagesByModel[selectedModel] = GeneratedImageSnapshot(
            image: generatedImage,
            generationTime: generationTime,
            memoryFootprintMB: memoryFootprintMB
        )
    }

    private func restoreGeneratedImage(for model: DiffusionModelKind) {
        if let snapshot = generatedImagesByModel[model] {
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

    func generate() async {
        guard !prompt.isEmpty, let resourceURL, isResourcesValid else { return }

        isGenerating = true
        errorMessage = nil
        let startTime = startGenerationTimer()

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
            let cgImage = try await Task.detached(priority: .userInitiated) {
                try autoreleasepool { () -> CGImage? in
                    let result = try pipeline.generateImages(
                        configuration: config,
                        progressHandler: { progress in
                            // Per-step memory probe so we can pinpoint OOM
                            // location: conditioning vs. step-N MMDiT vs. VAE decode.
                            MemoryProbe.log("step \(progress.step + 1)/\(progress.stepCount)")
                            return true
                        }
                    )
                    MemoryProbe.log("after MMDiT loop / before VAE decode")
                    return result.first.flatMap { $0 }
                }
            }.value
            MemoryProbe.log("after generateImages")
            logElapsed("generateImages", since: inferenceStart)

            let finalTime = CFAbsoluteTimeGetCurrent() - startTime
            stopGenerationTimer(finalTime: finalTime)
            memoryFootprintMB = MemoryProbe.residentMB()
            if let cgImage {
                let image = UIImage(cgImage: cgImage)
                generatedImage = image
                generatedImagesByModel[selectedModel] = GeneratedImageSnapshot(
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

                HStack(alignment: .center, spacing: 10) {
                    Picker(
                        "Diffusion steps",
                        selection: Binding(
                            get: { viewModel.selectedModel },
                            set: { viewModel.selectModel($0) }
                        )
                    ) {
                        ForEach(DiffusionModelKind.allCases) { model in
                            Text(model.selectorLabel).tag(model)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(viewModel.isGenerating)

                    Text(viewModel.activeResolutionLabel)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.green)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(Color(UIColor.secondarySystemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
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
                Button(action: {
                    Task { await viewModel.generate() }
                }) {
                    HStack(spacing: 8) {
                        Image(systemName: viewModel.isGenerating ? "hourglass" : "sparkles")
                        Text(viewModel.isGenerating ? "Generating" : "Generate")
                    }
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity, minHeight: 48)
                    .background(canGenerate ? Color.blue : Color.gray)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                .disabled(!canGenerate)

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

                    Text("Download SD3 Medium Core ML resources into this app. The files are stored locally and reused across launches.")
                        .font(.system(size: 14))
                        .foregroundColor(secondaryTextColor)
                }

                VStack(alignment: .leading, spacing: 12) {
                    ResourceStatusRow(
                        title: "2 steps",
                        isReady: resourceManager.hasResources(for: .sd3MediumTwoStep)
                    )
                    ResourceStatusRow(
                        title: "4 steps",
                        isReady: resourceManager.hasResources(for: .sd3MediumFourStep)
                    )
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
                            await resourceManager.downloadSelected(selectedModel)
                            onResourcesChanged()
                        }
                    } label: {
                        Label("Download Selected Model", systemImage: "arrow.down.circle")
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
                        Label("Download 2-Step and 4-Step", systemImage: "square.and.arrow.down")
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
