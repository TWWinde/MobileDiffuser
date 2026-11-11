// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

/// A decoder model which produces RGB images from latent samples
@available(iOS 16.2, macOS 13.1, *)
public struct Decoder: ResourceManaging {

    /// VAE decoder model
    var model: ManagedMLModel

    /// Create decoder from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled VAE decoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: A decoder that will lazily load its required resources when needed or requested
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }

    /// Ensure the model has been loaded into memory
    public func loadResources() throws {
        try model.loadResources()
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
        model.unloadResources()
    }

    /// Batch decode latent samples into images
    ///
    ///  - Parameters:
    ///    - latents: Batch of latent samples to decode
    ///    - scaleFactor: scalar divisor on latents before decoding
    ///  - Returns: decoded images
    public func decode(
        _ latents: [MLShapedArray<Float32>],
        scaleFactor: Float32,
        shiftFactor: Float32 = 0.0
    ) throws -> [CGImage] {

        // Latents larger than the compiled decoder input (e.g. 128x128 latents for 1024px with a
        // 64x64 / 512px decoder) are decoded as overlapping tiles and feather-blended.
        if let tileShape = inputShape, tileShape.count == 4,
           let first = latents.first, first.shape.count == 4,
           first.shape[2] > tileShape[2] || first.shape[3] > tileShape[3] {
            return try latents.map { latent in
                try decodeTiled(latent, tileH: tileShape[2], tileW: tileShape[3],
                                scaleFactor: scaleFactor, shiftFactor: shiftFactor)
            }
        }

        // Form batch inputs for model
        let inputs: [MLFeatureProvider] = try latents.map { sample in
            // Reference pipeline scales the latent samples before decoding
            let sampleScaled = MLShapedArray<Float32>(
                scalars: sample.scalars.map { $0 / scaleFactor + shiftFactor },
                shape: sample.shape)

            let dict = [inputName: MLMultiArray(sampleScaled)]
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)

        // Batch predict with model
        let results = try model.perform { model in
            try model.predictions(fromBatch: batch)
        }

        // Transform the outputs to CGImages
        let images: [CGImage] = try (0..<results.count).map { i in
            let result = results.features(at: i)
            let outputName = result.featureNames.first!
            let output = result.featureValue(for: outputName)!.multiArrayValue!
            return try CGImage.fromShapedArray(MLShapedArray<Float32>(converting: output))
        }

        return images
    }

    var inputName: String {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.key
        }
    }

    /// Static input shape of the compiled decoder, e.g. [1, 16, 64, 64].
    var inputShape: [Int]? {
        (try? model.perform { model in
            model.modelDescription.inputDescriptionsByName.first?.value
                .multiArrayConstraint?.shape.map { $0.intValue }
        }) ?? nil
    }

    /// Tile start offsets covering `length` with tiles of `tile`, overlapping by about half a tile.
    static func tileStarts(length: Int, tile: Int) -> [Int] {
        guard length > tile else { return [0] }
        let count = Int((Double(length - tile) / Double(tile / 2)).rounded(.up)) + 1
        return (0..<count).map { i in
            Int((Double(i) * Double(length - tile) / Double(count - 1)).rounded())
        }
    }

    /// 1D blend weight for pixel `x` in a tile of `size` pixels: linear ramp over `ramp` pixels on
    /// sides that overlap a neighbouring tile, flat (1) on image borders.
    static func featherWeight(_ x: Int, size: Int, ramp: Int, rampLow: Bool, rampHigh: Bool) -> Float {
        var w: Float = 1
        if rampLow { w = min(w, (Float(x) + 0.5) / Float(ramp)) }
        if rampHigh { w = min(w, (Float(size - 1 - x) + 0.5) / Float(ramp)) }
        return max(w, 1e-4)
    }

    func decodeTiled(
        _ latent: MLShapedArray<Float32>,
        tileH: Int, tileW: Int,
        scaleFactor: Float32, shiftFactor: Float32
    ) throws -> CGImage {
        let channels = latent.shape[1], height = latent.shape[2], width = latent.shape[3]
        let ys = Self.tileStarts(length: height, tile: tileH)
        let xs = Self.tileStarts(length: width, tile: tileW)
        let src = latent.scalars
        let name = inputName
        var accum: [Float32] = []
        var weightSum: [Float32] = []
        var scale = 0, outH = 0, outW = 0

        for (iy, y0) in ys.enumerated() {
            for (ix, x0) in xs.enumerated() {
                var tile = [Float32](repeating: 0, count: channels * tileH * tileW)
                for c in 0..<channels {
                    for y in 0..<tileH {
                        let srcRow = (c * height + y0 + y) * width + x0
                        let dstRow = (c * tileH + y) * tileW
                        for x in 0..<tileW {
                            tile[dstRow + x] = src[srcRow + x] / scaleFactor + shiftFactor
                        }
                    }
                }
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    name: MLMultiArray(MLShapedArray<Float32>(scalars: tile, shape: [1, channels, tileH, tileW]))
                ])
                let result = try model.perform { model in try model.prediction(from: input) }
                let outName = result.featureNames.first!
                let decoded = MLShapedArray<Float32>(converting: result.featureValue(for: outName)!.multiArrayValue!)
                let pixH = decoded.shape[2], pixW = decoded.shape[3]

                if accum.isEmpty {
                    scale = pixH / tileH
                    outH = height * scale; outW = width * scale
                    accum = [Float32](repeating: 0, count: 3 * outH * outW)
                    weightSum = [Float32](repeating: 0, count: outH * outW)
                }
                let rampY = (ys.count > 1 ? tileH - (ys[1] - ys[0]) : 1) * scale
                let rampX = (xs.count > 1 ? tileW - (xs[1] - xs[0]) : 1) * scale
                let wy = (0..<pixH).map { Self.featherWeight($0, size: pixH, ramp: max(rampY, 1),
                                                             rampLow: iy > 0, rampHigh: iy < ys.count - 1) }
                let wx = (0..<pixW).map { Self.featherWeight($0, size: pixW, ramp: max(rampX, 1),
                                                             rampLow: ix > 0, rampHigh: ix < xs.count - 1) }
                let pix = decoded.scalars
                let oy = y0 * scale, ox = x0 * scale
                let plane = outH * outW
                for y in 0..<pixH {
                    for x in 0..<pixW {
                        let w = wy[y] * wx[x]
                        let o = (oy + y) * outW + ox + x
                        weightSum[o] += w
                        for c in 0..<3 {
                            accum[c * plane + o] += w * pix[(c * pixH + y) * pixW + x]
                        }
                    }
                }
            }
        }

        let plane = outH * outW
        for c in 0..<3 {
            for i in 0..<plane { accum[c * plane + i] /= weightSum[i] }
        }
        return try CGImage.fromShapedArray(MLShapedArray<Float32>(scalars: accum, shape: [1, 3, outH, outW]))
    }

}
