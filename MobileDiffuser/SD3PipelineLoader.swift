import CoreML
import Foundation
import StableDiffusion

enum SD3PipelineLoader {
    enum LoadError: LocalizedError {
        case modelNotFound(String)
        case insufficientMemory(requiredGB: Double, availableGB: Double)
        case allComputeUnitsFailed(String)

        var errorDescription: String? {
            switch self {
            case .modelNotFound(let name):
                return "缺少模型文件: \(name)"
            case .insufficientMemory(let requiredGB, let availableGB):
                return String(
                    format: "模型需要约 %.1f GB 内存，设备可用约 %.1f GB。",
                    requiredGB, availableGB
                )
            case .allComputeUnitsFailed(let detail):
                return "所有 compute unit 都无法加载: \(detail)"
            }
        }
    }

    /// Compute-unit profile, ordered fastest → most-compatible.
    /// The bundled two-step SD3 MMDiT is split into several Core ML stages and
    /// uses INT8 linear weight quantization. The split keeps ANE compile/load
    /// pressure below the per-model limit, while INT8 avoids fp16 expansion of
    /// the large transformer weights at load time.
    enum ComputeUnitsProfile: String, CaseIterable {
        case aneFirst   // .cpuAndNeuralEngine   — preferred: weights live on ANE
        case hybrid     // Text/VAE on ANE, MMDiT on GPU — fastest stable path for current MMDiT
        case gpuFirst   // .cpuAndGPU            — fallback if ANE init fails
        case cpuOnly    // .cpuOnly              — slow but always works

        var coreML: MLComputeUnits {
            switch self {
            case .cpuOnly:  return .cpuOnly
            case .gpuFirst, .hybrid: return .cpuAndGPU
            case .aneFirst: return .cpuAndNeuralEngine
            }
        }
    }

    static func resolveResourceURL(
        in bundle: Bundle = .main,
        folderName: String = DiffusionModelKind.sd3MediumTwoStep.resourceFolderName
    ) -> URL? {
        if let folderURL = bundle.url(forResource: folderName, withExtension: nil) {
            return folderURL
        }
        if folderName == DiffusionModelKind.sd3MediumTwoStep.resourceFolderName,
           let rootURL = bundle.resourceURL,
           hasRequiredResources(at: rootURL)
        {
            return rootURL
        }
        if folderName == DiffusionModelKind.sd3MediumTwoStep.resourceFolderName,
           let mmditURL = bundle.url(forResource: "MultiModalDiffusionTransformer", withExtension: "mlmodelc")
        {
            return mmditURL.deletingLastPathComponent()
        }
        return nil
    }

    static func hasRequiredResources(at url: URL) -> Bool {
        missingResources(at: url).isEmpty
    }

    static func missingResources(at url: URL) -> [String] {
        let fm = FileManager.default
        let requiredNames = [
            "TextEncoder.mlmodelc",
            "TextEncoder2.mlmodelc",
            "VAEDecoder.mlmodelc",
            "vocab.json",
            "merges.txt",
            "MultiModalDiffusionTransformerConditioning.mlmodelc",
            "MultiModalDiffusionTransformerStage0.mlmodelc",
        ]
        return requiredNames.filter {
            !fm.fileExists(atPath: url.appending(path: $0).path)
        }
    }

    /// Build a pipeline at the given compute-units profile.
    /// `prewarm` is intentionally disabled for the iPhone path; eager ANE plan
    /// compilation spikes memory before generation has a chance to stream the
    /// split MMDiT stages one at a time.
    static func createPipeline(
        at resourceURL: URL,
        profile: ComputeUnitsProfile,
        prewarm: Bool = false
    ) throws -> StableDiffusion3Pipeline {
        let urls = StableDiffusion3Pipeline.ResourceURLs(resourcesAt: resourceURL)

        let hasSplitMMDiT = FileManager.default.fileExists(atPath: urls.mmditConditioningURL.path)
            && !urls.mmditStageURLs.isEmpty

        let required: [(String, URL)] = [
            ("TextEncoder", urls.textEncoderURL),
            ("TextEncoder2", urls.textEncoder2URL),
            ("VAEDecoder", urls.decoderURL),
            ("vocab.json", urls.vocabURL),
            ("merges.txt", urls.mergesURL),
        ]
        for (name, url) in required where !FileManager.default.fileExists(atPath: url.path) {
            throw LoadError.modelNotFound(name)
        }
        if hasSplitMMDiT {
            let adalnLayout: String
            if urls.mmditAdaptiveLayerNormStageURLs.count == urls.mmditStageURLs.count {
                adalnLayout = "\(urls.mmditAdaptiveLayerNormStageURLs.count) AdaLN stages"
            } else if FileManager.default.fileExists(atPath: urls.mmditAdaptiveLayerNormURL.path) {
                adalnLayout = "single AdaLN"
            } else {
                adalnLayout = "fused AdaLN"
            }
            print("[SD3] using split MMDiT: \(urls.mmditStageURLs.count) stages + \(adalnLayout)")
        } else if profile == .aneFirst {
            throw LoadError.modelNotFound(
                "split MMDiT resources: MultiModalDiffusionTransformerConditioning/Stage0...StageN.mlmodelc"
            )
        } else if !FileManager.default.fileExists(atPath: urls.mmditURL.path) {
            throw LoadError.modelNotFound("MultiModalDiffusionTransformer")
        }

        let units = profile.coreML
        print("[SD3] Building pipeline with profile=\(profile.rawValue) (\(units.description))")

        var mmditConfig: MLModelConfiguration
        var mmditPrecomputeConfig: MLModelConfiguration
        var textEncoderConfig: MLModelConfiguration
        var textEncoder2Config: MLModelConfiguration
        var decoderConfig: MLModelConfiguration

        if profile == .hybrid {
            // CLIP/VAE still benefit from ANE if a future fallback uses GPU for
            // the large MMDiT body, so keep those on ANE in the hybrid profile.
            mmditConfig        = makeConfig(.cpuAndGPU)
            mmditPrecomputeConfig = makeConfig(.cpuOnly)
            textEncoderConfig  = makeConfig(.cpuAndNeuralEngine)
            textEncoder2Config = makeConfig(.cpuAndNeuralEngine)
            decoderConfig      = makeConfig(.cpuAndNeuralEngine)
        } else {
            // Same compute units across all sub-models. CoreML will internally
            // route incompatible ops to CPU within the same plan.
            mmditConfig        = makeConfig(units)
            // Conditioning/AdaLN are precomputed once per run. Keep them off
            // ANE so the large AdaLN helper does not consume ANE compiler
            // budget before the actual MMDiT body stages are loaded.
            mmditPrecomputeConfig = makeConfig(.cpuOnly)
            textEncoderConfig  = makeConfig(units)
            textEncoder2Config = makeConfig(units)
            decoderConfig      = makeConfig(units)
        }

        let tokenizer = try BPETokenizer(mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL)
        let textEncoder = TextEncoderXL(
            tokenizer: tokenizer, modelAt: urls.textEncoderURL, configuration: textEncoderConfig
        )

        let tokenizer2 = try BPETokenizer(
            mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL, padToken: "!"
        )
        let textEncoder2 = TextEncoderXL(
            tokenizer: tokenizer2, modelAt: urls.textEncoder2URL, configuration: textEncoder2Config
        )

        let mmdit: MultiModalDiffusionTransformer
        if hasSplitMMDiT {
            if urls.mmditAdaptiveLayerNormStageURLs.count == urls.mmditStageURLs.count {
                mmdit = MultiModalDiffusionTransformer(
                    conditioningAt: urls.mmditConditioningURL,
                    conditioningConfiguration: mmditPrecomputeConfig,
                    adaptiveLayerNormStagesAt: urls.mmditAdaptiveLayerNormStageURLs,
                    adaptiveLayerNormConfiguration: mmditPrecomputeConfig,
                    stagesAt: urls.mmditStageURLs,
                    stagesConfiguration: mmditConfig
                )
            } else if FileManager.default.fileExists(atPath: urls.mmditAdaptiveLayerNormURL.path) {
                mmdit = MultiModalDiffusionTransformer(
                    conditioningAt: urls.mmditConditioningURL,
                    conditioningConfiguration: mmditPrecomputeConfig,
                    adaptiveLayerNormAt: urls.mmditAdaptiveLayerNormURL,
                    adaptiveLayerNormConfiguration: mmditPrecomputeConfig,
                    stagesAt: urls.mmditStageURLs,
                    stagesConfiguration: mmditConfig
                )
            } else {
                mmdit = MultiModalDiffusionTransformer(
                    conditioningAt: urls.mmditConditioningURL,
                    conditioningConfiguration: mmditPrecomputeConfig,
                    stagesAt: urls.mmditStageURLs,
                    stagesConfiguration: mmditConfig
                )
            }
        } else {
            mmdit = MultiModalDiffusionTransformer(modelAt: urls.mmditURL, configuration: mmditConfig)
        }
        let decoder = Decoder(modelAt: urls.decoderURL, configuration: decoderConfig)

        let pipeline = StableDiffusion3Pipeline(
            textEncoder: textEncoder,
            textEncoder2: textEncoder2,
            textEncoderT5: nil,
            mmdit: mmdit,
            decoder: decoder,
            encoder: nil,
            reduceMemory: true
        )

        if prewarm {
            MemoryProbe.log("before prewarm[\(profile.rawValue)]")
            try pipeline.prewarmResources()
            MemoryProbe.log("after  prewarm[\(profile.rawValue)]")
        }

        return pipeline
    }

    /// Try compute units in order until one pipeline is constructed.
    /// Returns the working pipeline plus the profile that won.
    /// `gpuMinBudgetMB` guards against GPU fallback on memory-starved devices.
    /// ANE is still attempted below this threshold because the large compressed
    /// weights are not charged to app dirty memory the same way GPU pages are.
    static func createPipelineWithFallback(
        at resourceURL: URL,
        profiles: [ComputeUnitsProfile] = [.aneFirst],
        gpuMinBudgetMB: Double = 2500
    ) throws -> (StableDiffusion3Pipeline, ComputeUnitsProfile) {
        let availableMB = MemoryProbe.availableMB()
        print(String(format: "[SD3] memory budget: %.0f MB available", availableMB))
        print("[SD3] requested fallback order: \(profiles.map(\.rawValue).joined(separator: " -> "))")

        let effective: [ComputeUnitsProfile]
        if availableMB > 0, availableMB < gpuMinBudgetMB {
            effective = profiles.filter { $0 != .gpuFirst && $0 != .hybrid }
            print("[SD3] insufficient budget (<\(Int(gpuMinBudgetMB))MB) — skipping GPU fallback")
        } else {
            effective = profiles
        }
        print("[SD3] effective fallback order: \(effective.map(\.rawValue).joined(separator: " -> "))")

        var lastError: Error?
        for profile in effective {
            do {
                let pipeline = try createPipeline(at: resourceURL, profile: profile, prewarm: false)
                print("[SD3] Pipeline ready with profile=\(profile.rawValue)")
                return (pipeline, profile)
            } catch {
                print("[SD3] profile=\(profile.rawValue) failed: \(error.localizedDescription)")
                lastError = error
            }
        }
        throw LoadError.allComputeUnitsFailed(lastError?.localizedDescription ?? "unknown")
    }

    private static func makeConfig(_ units: MLComputeUnits) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = units
        // Keep memoryReduction defaults; the StableDiffusion3Pipeline already
        // handles per-stage load/unload via its `reduceMemory` flag.
        return config
    }
}

private extension MLComputeUnits {
    var description: String {
        switch self {
        case .cpuOnly:              return "cpuOnly"
        case .cpuAndGPU:            return "cpuAndGPU"
        case .cpuAndNeuralEngine:   return "cpuAndNeuralEngine"
        case .all:                  return "all"
        @unknown default:           return "unknown(\(rawValue))"
        }
    }
}
