// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation

import CoreML

private struct MMDiTArrayStats {
    var count: Int = 0
    var finiteCount: Int = 0
    var nanCount: Int = 0
    var infiniteCount: Int = 0
    var minimum: Float = .greatestFiniteMagnitude
    var maximum: Float = -.greatestFiniteMagnitude
    var maxAbs: Float = 0
    var sum: Double = 0

    var mean: Float {
        finiteCount > 0 ? Float(sum / Double(finiteCount)) : .nan
    }

    var hasNonFinite: Bool {
        nanCount > 0 || infiniteCount > 0
    }
}

private struct NonFiniteMMDiTOutputError: LocalizedError {
    let label: String
    let stats: MMDiTArrayStats

    var errorDescription: String? {
        "\(label) contains non-finite values (nan=\(stats.nanCount), inf=\(stats.infiniteCount), count=\(stats.count))"
    }
}

private let enableMMDiTArrayDiagnostics =
    ProcessInfo.processInfo.environment["SD3_MMDIT_DIAGNOSTICS"] == "1"

private func mmditStats(_ array: MLMultiArray) -> MMDiTArrayStats {
    var stats = MMDiTArrayStats()
    stats.count = array.count

    func observe(_ value: Float) {
        if value.isNaN {
            stats.nanCount += 1
            return
        }
        if !value.isFinite {
            stats.infiniteCount += 1
            return
        }
        stats.finiteCount += 1
        stats.minimum = min(stats.minimum, value)
        stats.maximum = max(stats.maximum, value)
        stats.maxAbs = max(stats.maxAbs, abs(value))
        stats.sum += Double(value)
    }

    switch array.dataType {
    case .float16:
        let pointer = array.dataPointer.assumingMemoryBound(to: UInt16.self)
        for index in 0..<array.count {
            observe(Float(Float16(bitPattern: pointer[index])))
        }
    case .float32:
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        for index in 0..<array.count {
            observe(pointer[index])
        }
    case .double:
        let pointer = array.dataPointer.assumingMemoryBound(to: Double.self)
        for index in 0..<array.count {
            observe(Float(pointer[index]))
        }
    case .int32:
        let pointer = array.dataPointer.assumingMemoryBound(to: Int32.self)
        for index in 0..<array.count {
            observe(Float(pointer[index]))
        }
    @unknown default:
        for index in 0..<array.count {
            observe(array[index].floatValue)
        }
    }

    return stats
}

private func logAndValidateMMDiTArray(_ array: MLMultiArray, label: String) throws {
    guard enableMMDiTArrayDiagnostics else { return }

    let stats = mmditStats(array)
    let minValue = stats.finiteCount > 0 ? stats.minimum : .nan
    let maxValue = stats.finiteCount > 0 ? stats.maximum : .nan
    print(
        "[SD3] \(label): count=\(stats.count) mean=\(stats.mean) min=\(minValue) max=\(maxValue) maxAbs=\(stats.maxAbs) nan=\(stats.nanCount) inf=\(stats.infiniteCount)"
    )
    if stats.hasNonFinite {
        throw NonFiniteMMDiTOutputError(label: label, stats: stats)
    }
}

/// MMDiT noise prediction model for stable diffusion
@available(iOS 16.2, macOS 13.1, *)
public struct MultiModalDiffusionTransformer: ResourceManaging {
    struct PrecomputedConditioning {
        let modulationInputs: MLMultiArray?
        let adaptiveLayerNormFeatures: [String: MLFeatureValue]?
    }

    enum Backend {
        case single(models: [ManagedMLModel])
        case chunked(
            conditioning: ManagedMLModel,
            adaptiveLayerNorm: ManagedMLModel?,
            adaptiveLayerNormStages: [ManagedMLModel],
            stages: [ManagedMLModel]
        )

        var bodyModels: [ManagedMLModel] {
            switch self {
            case .single(let models):
                return models
            case .chunked(_, _, _, let stages):
                return stages
            }
        }

        var allModels: [ManagedMLModel] {
            switch self {
            case .single(let models):
                return models
            case .chunked(let conditioning, let adaptiveLayerNorm, let adaptiveLayerNormStages, let stages):
                return [conditioning] + [adaptiveLayerNorm].compactMap { $0 } + adaptiveLayerNormStages + stages
            }
        }
    }

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    ///
    /// It can be in the form of a single model or multiple stages
    var models: [ManagedMLModel]
    var backend: Backend

    /// Creates a MMDiT noise prediction model
    ///
    /// - Parameters:
    ///   - url: Location of single MMDiT compiled Core ML model
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: MMDiT model that will lazily load its required resources when needed or requested
    public init(modelAt url: URL,
                configuration: MLModelConfiguration)
    {
        self.models = [ManagedMLModel(modelAt: url, configuration: configuration)]
        self.backend = .single(models: self.models)
    }

    /// Creates a split MMDiT noise prediction model.
    ///
    /// The split layout is:
    /// - Conditioning: pooled text + timestep -> modulation inputs
    /// - StageN: serial fused MMDiT body chunks, consuming modulation inputs directly
    public init(
        conditioningAt conditioningURL: URL,
        stagesAt stageURLs: [URL],
        configuration: MLModelConfiguration
    ) {
        let conditioning = ManagedMLModel(modelAt: conditioningURL, configuration: configuration)
        let stages = stageURLs.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
        self.models = stages
        self.backend = .chunked(
            conditioning: conditioning,
            adaptiveLayerNorm: nil,
            adaptiveLayerNormStages: [],
            stages: stages
        )
    }

    /// Creates a legacy split MMDiT noise prediction model.
    ///
    /// The legacy split layout is:
    /// - Conditioning: pooled text + timestep -> modulation inputs
    /// - AdaptiveLayerNorm: modulation inputs -> all AdaLN tensors
    /// - StageN: serial MMDiT body chunks, consuming the AdaLN tensors they need
    public init(
        conditioningAt conditioningURL: URL,
        adaptiveLayerNormAt adaptiveLayerNormURL: URL,
        stagesAt stageURLs: [URL],
        configuration: MLModelConfiguration
    ) {
        let conditioning = ManagedMLModel(modelAt: conditioningURL, configuration: configuration)
        let adaptiveLayerNorm = ManagedMLModel(modelAt: adaptiveLayerNormURL, configuration: configuration)
        let stages = stageURLs.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
        self.models = stages
        self.backend = .chunked(
            conditioning: conditioning,
            adaptiveLayerNorm: adaptiveLayerNorm,
            adaptiveLayerNormStages: [],
            stages: stages
        )
    }

    /// Creates a split MMDiT noise prediction model with AdaLN split per body
    /// stage. This avoids a giant AdaLN helper with too many live outputs for
    /// the ANE compiler.
    public init(
        conditioningAt conditioningURL: URL,
        adaptiveLayerNormStagesAt adaptiveLayerNormStageURLs: [URL],
        stagesAt stageURLs: [URL],
        configuration: MLModelConfiguration
    ) {
        let conditioning = ManagedMLModel(modelAt: conditioningURL, configuration: configuration)
        let adaptiveLayerNormStages = adaptiveLayerNormStageURLs.map {
            ManagedMLModel(modelAt: $0, configuration: configuration)
        }
        let stages = stageURLs.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
        self.models = stages
        self.backend = .chunked(
            conditioning: conditioning,
            adaptiveLayerNorm: nil,
            adaptiveLayerNormStages: adaptiveLayerNormStages,
            stages: stages
        )
    }

    /// Creates a split MMDiT noise prediction model with separate compute-unit
    /// choices for precomputed conditioning and fused ANE body stages.
    public init(
        conditioningAt conditioningURL: URL,
        conditioningConfiguration: MLModelConfiguration,
        stagesAt stageURLs: [URL],
        stagesConfiguration: MLModelConfiguration
    ) {
        let conditioning = ManagedMLModel(modelAt: conditioningURL, configuration: conditioningConfiguration)
        let stages = stageURLs.map { ManagedMLModel(modelAt: $0, configuration: stagesConfiguration) }
        self.models = stages
        self.backend = .chunked(
            conditioning: conditioning,
            adaptiveLayerNorm: nil,
            adaptiveLayerNormStages: [],
            stages: stages
        )
    }

    /// Creates a split MMDiT noise prediction model with separate compute-unit
    /// choices for precomputed conditioning and legacy ANE body stages.
    public init(
        conditioningAt conditioningURL: URL,
        conditioningConfiguration: MLModelConfiguration,
        adaptiveLayerNormAt adaptiveLayerNormURL: URL,
        adaptiveLayerNormConfiguration: MLModelConfiguration,
        stagesAt stageURLs: [URL],
        stagesConfiguration: MLModelConfiguration
    ) {
        let conditioning = ManagedMLModel(modelAt: conditioningURL, configuration: conditioningConfiguration)
        let adaptiveLayerNorm = ManagedMLModel(modelAt: adaptiveLayerNormURL, configuration: adaptiveLayerNormConfiguration)
        let stages = stageURLs.map { ManagedMLModel(modelAt: $0, configuration: stagesConfiguration) }
        self.models = stages
        self.backend = .chunked(
            conditioning: conditioning,
            adaptiveLayerNorm: adaptiveLayerNorm,
            adaptiveLayerNormStages: [],
            stages: stages
        )
    }

    /// Creates a split MMDiT model with separate compute-unit choices and
    /// AdaLN split per body stage.
    public init(
        conditioningAt conditioningURL: URL,
        conditioningConfiguration: MLModelConfiguration,
        adaptiveLayerNormStagesAt adaptiveLayerNormStageURLs: [URL],
        adaptiveLayerNormConfiguration: MLModelConfiguration,
        stagesAt stageURLs: [URL],
        stagesConfiguration: MLModelConfiguration
    ) {
        let conditioning = ManagedMLModel(modelAt: conditioningURL, configuration: conditioningConfiguration)
        let adaptiveLayerNormStages = adaptiveLayerNormStageURLs.map {
            ManagedMLModel(modelAt: $0, configuration: adaptiveLayerNormConfiguration)
        }
        let stages = stageURLs.map { ManagedMLModel(modelAt: $0, configuration: stagesConfiguration) }
        self.models = stages
        self.backend = .chunked(
            conditioning: conditioning,
            adaptiveLayerNorm: nil,
            adaptiveLayerNormStages: adaptiveLayerNormStages,
            stages: stages
        )
    }

    /// Load resources.
    public func loadResources() throws {
        for model in backend.allModels {
            try model.loadResources()
        }
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
        for model in backend.allModels {
            model.unloadResources()
        }
    }

    /// Pre-warm resources
    public func prewarmResources() throws {
        // Override default to pre-warm each model
        for model in backend.allModels {
            print("[SD3] prewarm: \(model.name) begin")
            try model.loadResources()
            print("[SD3] prewarm: \(model.name) done")
            model.unloadResources()
        }
    }

    var latentImageEmbeddingsDescription: MLFeatureDescription {
        try! backend.bodyModels.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["latent_image_embeddings"]!
        }
    }

    /// The expected shape of the models latent sample input
    public var latentImageEmbeddingsShape: [Int] {
        latentImageEmbeddingsDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

    var tokenLevelTextEmbeddingsDescription: MLFeatureDescription {
        try! backend.bodyModels.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["token_level_text_embeddings"]!
        }
    }

    var usesSplitBody: Bool {
        if case .chunked = backend {
            return true
        }
        return false
    }

    func precomputeConditioning(
        timeSteps: [Float],
        batchSize: Int,
        pooledTextEmbeddings: MLShapedArray<Float32>
    ) throws -> [PrecomputedConditioning]? {
        guard case .chunked(let conditioning, let adaptiveLayerNorm, let adaptiveLayerNormStages, _) = backend else {
            return nil
        }
        defer {
            conditioning.unloadResources()
            adaptiveLayerNorm?.unloadResources()
            for model in adaptiveLayerNormStages {
                model.unloadResources()
            }
        }

        return try timeSteps.map { timeStep in
            let t = MLShapedArray<Float32>(
                scalars: Array(repeating: timeStep, count: batchSize),
                shape: [batchSize]
            )
            let conditioningInput = try MLDictionaryFeatureProvider(dictionary: [
                "pooled_text_embeddings": MLMultiArray(pooledTextEmbeddings),
                "timestep": MLMultiArray(t),
            ])

            let conditioningOutput = try conditioning.perform { model in
                try model.prediction(from: conditioningInput)
            }
            guard let modulationInputs = conditioningOutput
                .featureValue(for: "modulation_inputs")?
                .multiArrayValue
            else {
                throw PipelineError.missingMMDiTOutputs
            }
            try logAndValidateMMDiTArray(
                modulationInputs,
                label: "conditioning timestep=\(timeStep) modulation_inputs"
            )

            if let adaptiveLayerNorm {
                let adaptiveInput = try MLDictionaryFeatureProvider(dictionary: [
                    "modulation_inputs": modulationInputs,
                ])
                let adaptiveOutput = try adaptiveLayerNorm.perform { model in
                    try model.prediction(from: adaptiveInput)
                }

                return PrecomputedConditioning(
                    modulationInputs: nil,
                    adaptiveLayerNormFeatures: adaptiveOutput.featureValueDictionary
                )
            }

            return PrecomputedConditioning(
                modulationInputs: modulationInputs,
                adaptiveLayerNormFeatures: nil
            )
        }
    }

    /// The expected shape of the geometry conditioning
    public var tokenLevelTextEmbeddingsShape: [Int] {
        tokenLevelTextEmbeddingsDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

    /// Batch prediction noise from latent samples
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    /// - Returns: Array of predicted noise residuals
    func predictNoise(
        latents: [MLShapedArray<Float32>],
        timeStep: Float,
        tokenLevelTextEmbeddings: MLShapedArray<Float32>,
        pooledTextEmbeddings: MLShapedArray<Float32>,
        precomputedConditioning: PrecomputedConditioning? = nil
    ) throws -> [MLShapedArray<Float32>] {
        if case .chunked = backend {
            return try predictNoiseWithSplitBody(
                latents: latents,
                timeStep: timeStep,
                tokenLevelTextEmbeddings: tokenLevelTextEmbeddings,
                pooledTextEmbeddings: pooledTextEmbeddings,
                precomputedConditioning: precomputedConditioning
            )
        }

        // Match time step batch dimension to the actual latent batch (1 for
        // single-conditional / distilled inference, 2 for classic CFG).
        let batchSize = latents.first?.shape.first ?? 2
        let t = MLShapedArray<Float32>(
            scalars: Array(repeating: timeStep, count: batchSize),
            shape: [batchSize]
        )

        // Form batch input to model
        let inputs = try latents.enumerated().map {
            let dict: [String: Any] = [
                "latent_image_embeddings": MLMultiArray($0.element),
                "timestep": MLMultiArray(t),
                "token_level_text_embeddings": MLMultiArray(tokenLevelTextEmbeddings),
                "pooled_text_embeddings": MLMultiArray(pooledTextEmbeddings),
            ]
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)

        // Make predictions
        let results = try models.predictions(from: batch)

        // Pull out the results in Float32 format
        let noise = (0..<results.count).map { i in

            let result = results.features(at: i)
            let outputName = result.featureNames.first!

            let outputNoise = result.featureValue(for: outputName)!.multiArrayValue!

            // To conform to this func return type make sure we return float32
            // Use the fact that the concatenating constructor for MLMultiArray
            // can do type conversion:
            let fp32Noise = MLMultiArray(
                concatenating: [outputNoise],
                axis: 0,
                dataType: .float32
            )
            return MLShapedArray<Float32>(fp32Noise)
        }

        return noise
    }

    private func predictNoiseWithSplitBody(
        latents: [MLShapedArray<Float32>],
        timeStep: Float,
        tokenLevelTextEmbeddings: MLShapedArray<Float32>,
        pooledTextEmbeddings: MLShapedArray<Float32>,
        precomputedConditioning: PrecomputedConditioning?
    ) throws -> [MLShapedArray<Float32>] {
        guard case .chunked(_, _, let adaptiveLayerNormStages, let stages) = backend else {
            return []
        }

        let batchSize = latents.first?.shape.first ?? 2
        let conditioning: PrecomputedConditioning
        if let precomputedConditioning {
            conditioning = precomputedConditioning
        } else {
            conditioning = try precomputeConditioning(
                timeSteps: [timeStep],
                batchSize: batchSize,
                pooledTextEmbeddings: pooledTextEmbeddings
            )!.first!
        }

        return try latents.map { latent in
            var features = conditioning.adaptiveLayerNormFeatures?.reduce(into: [String: Any]()) {
                $0[$1.key] = $1.value
            } ?? [:]
            if let modulationInputs = conditioning.modulationInputs {
                features["modulation_inputs"] = modulationInputs
            }
            features["latent_image_embeddings"] = MLMultiArray(latent)
            features["token_level_text_embeddings"] = MLMultiArray(tokenLevelTextEmbeddings)

            var result: MLFeatureProvider?
            for (stageIndex, stage) in stages.enumerated() {
                if !adaptiveLayerNormStages.isEmpty {
                    guard let modulationInputs = conditioning.modulationInputs else {
                        throw PipelineError.missingMMDiTOutputs
                    }
                    let adaptiveInput = try MLDictionaryFeatureProvider(dictionary: [
                        "modulation_inputs": modulationInputs,
                    ])
                    let adaptiveFeatures = try adaptiveLayerNormStages[stageIndex].perform { model in
                        try model.prediction(from: adaptiveInput)
                    }
                    adaptiveLayerNormStages[stageIndex].unloadResources()
                    for name in adaptiveFeatures.featureNames {
                        features[name] = adaptiveFeatures.featureValue(for: name)
                    }
                }

                do {
                    result = try stage.perform { model in
                        let acceptedInputNames = Set(model.modelDescription.inputDescriptionsByName.keys)
                        let stageFeatures = features.filter { acceptedInputNames.contains($0.key) }
                        let provider = try MLDictionaryFeatureProvider(dictionary: stageFeatures)
                        return try model.prediction(from: provider)
                    }
                } catch {
                    stage.unloadResources()
                    throw error
                }
                stage.unloadResources()

                guard let result else {
                    throw PipelineError.missingMMDiTOutputs
                }

                for name in result.featureNames.sorted() {
                    guard let multiArray = result.featureValue(for: name)?.multiArrayValue else {
                        continue
                    }
                    try logAndValidateMMDiTArray(
                        multiArray,
                        label: "stage \(stageIndex) \(stage.name) output \(name)"
                    )
                }

                if stageIndex < stages.count - 1 {
                    for name in result.featureNames {
                        guard let value = result.featureValue(for: name) else {
                            continue
                        }
                        if let multiArray = value.multiArrayValue {
                            features[name] = multiArray
                            if name == "latent_image_embeddings_out" {
                                features["latent_image_embeddings"] = multiArray
                            } else if name == "token_level_text_embeddings_out" {
                                features["token_level_text_embeddings"] = multiArray
                            }
                        } else {
                            features[name] = value
                        }
                    }

                    if result.featureNames.contains("latent_image_embeddings_out") {
                        let exactTransientNames = [
                            "image_q",
                            "image_k",
                            "image_v",
                            "text_q",
                            "text_k",
                            "text_v",
                            "image_sdpa_output",
                            "text_sdpa_output",
                            "image_post_attn_scale",
                            "image_post_norm2_shift",
                            "image_post_norm2_residual_scale",
                            "image_post_mlp_scale",
                            "text_post_attn_scale",
                            "text_post_norm2_shift",
                            "text_post_norm2_residual_scale",
                            "text_post_mlp_scale",
                        ]
                        for transientName in exactTransientNames {
                            features.removeValue(forKey: transientName)
                        }
                        let chunkTransientNames = features.keys.filter {
                            $0.hasPrefix("image_sdpa_output_")
                        }
                        for transientName in chunkTransientNames {
                            features.removeValue(forKey: transientName)
                        }
                    }
                }
            }

            guard
                let outputName = result?.featureNames.first,
                let outputNoise = result?.featureValue(for: outputName)?.multiArrayValue
            else {
                throw PipelineError.missingMMDiTOutputs
            }

            let fp32Noise = MLMultiArray(
                concatenating: [outputNoise],
                axis: 0,
                dataType: .float32
            )
            return MLShapedArray<Float32>(fp32Noise)
        }
    }
}
