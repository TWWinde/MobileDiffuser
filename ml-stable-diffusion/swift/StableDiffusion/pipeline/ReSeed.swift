// ReSeed first-interval residual biases for SD3 joint blocks.
// https://twwinde.github.io/reseed-project-page/
//
// For block b the adapter adds a learned per-channel bias to the complete block output of the
// image stream (and of the text stream, except for the last, context-pre-only block). It is active
// only for the first sampler interval; later steps use the unchanged student.
//
// On ANE this requires one transformer block per Core ML stage, so every block output is a stage
// output: biases for blocks 0..22 are added here between stages, and the last block's image bias is
// fed to the final stage as the `reseed_image_bias` input (zeros when inactive).

import Accelerate
import CoreML
import Foundation

@available(iOS 16.2, macOS 13.1, *)
public struct ReSeedBiases {
    /// Per-block image-stream bias (count = number of transformer blocks).
    public let image: [[Float]]
    /// Per-block text-stream bias; nil for the context-pre-only last block.
    public let context: [[Float]?]

    public static let resourceName = "ReSeedBiases.json"

    /// Time spent adding biases between stages during the current denoiser call (diagnostic).
    nonisolated(unsafe) static var addSeconds: Double = 0

    private struct Payload: Decodable {
        let image: [[Float]]
        let context: [[Float]?]
    }

    public init(contentsOf url: URL) throws {
        let payload = try JSONDecoder().decode(Payload.self, from: Data(contentsOf: url))
        self.image = payload.image
        self.context = payload.context
    }

    /// Diagnostic switch (env MD_RESEED_DIAG): "skip" = do nothing, "touch" = only obtain the data pointer,
    /// "slow" = the original element-wise loop; default = fast path.
    static let diag = ProcessInfo.processInfo.environment["MD_RESEED_DIAG"] ?? ""

    /// Adds `scale * bias` to every token of a [1, tokens, channels] Core ML array in place.
    static func add(_ bias: [Float], scale: Float, to array: MLMultiArray) {
        if diag == "skip" { return }
        if diag == "touch" { _ = array.dataPointer; return }
        if diag != "slow" {
            let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
            if shape.count == 3, shape[2] == bias.count, strides[2] == 1 {
                let tokens = shape[1], channels = shape[2], tokenStride = strides[1]
                if array.dataType == .float16 {
                    // Same arithmetic as the element-wise loop (widen to Float, add, round back to
                    // Float16), but one vectorised conversion / add per token row.
                    let b = bias.map { scale * $0 }
                    nonisolated(unsafe) let p = array.dataPointer.assumingMemoryBound(to: Float16.self)
                    // The cost is memory traffic over the whole tensor, so split the token rows across cores.
                    let chunks = diag == "serial" ? 1 : Swift.min(8, Swift.max(1, tokens / 256))
                    let per = (tokens + chunks - 1) / chunks
                    DispatchQueue.concurrentPerform(iterations: chunks) { k in
                        var tmp = [Float](repeating: 0, count: channels)
                        tmp.withUnsafeMutableBufferPointer { tp in
                            for t in (k * per)..<Swift.min(tokens, (k + 1) * per) {
                                var half = vImage_Buffer(data: p + t * tokenStride, height: 1,
                                                         width: vImagePixelCount(channels), rowBytes: channels * 2)
                                var full = vImage_Buffer(data: tp.baseAddress!, height: 1,
                                                         width: vImagePixelCount(channels), rowBytes: channels * 4)
                                vImageConvert_Planar16FtoPlanarF(&half, &full, 0)
                                vDSP_vadd(tp.baseAddress!, 1, b, 1, tp.baseAddress!, 1, vDSP_Length(channels))
                                vImageConvert_PlanarFtoPlanar16F(&full, &half, 0)
                            }
                        }
                    }
                    return
                }
                if array.dataType == .float32 {
                    let b = bias.map { scale * $0 }
                    let p = array.dataPointer.assumingMemoryBound(to: Float.self)
                    for t in 0..<tokens { vDSP_vadd(p + t * tokenStride, 1, b, 1, p + t * tokenStride, 1, vDSP_Length(channels)) }
                    return
                }
            }
        }
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard shape.count == 3, shape[2] == bias.count else { return }
        let tokens = shape[1], channels = shape[2]
        let tokenStride = strides[1], channelStride = strides[2]
        switch array.dataType {
        case .float16:
            let p = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for t in 0..<tokens {
                let row = t * tokenStride
                for c in 0..<channels {
                    let i = row + c * channelStride
                    p[i] = Float16(Float(p[i]) + scale * bias[c])
                }
            }
        case .float32:
            let p = array.dataPointer.assumingMemoryBound(to: Float.self)
            for t in 0..<tokens {
                let row = t * tokenStride
                for c in 0..<channels {
                    p[row + c * channelStride] += scale * bias[c]
                }
            }
        default:
            for t in 0..<tokens {
                for c in 0..<channels {
                    let idx = [0, t, c] as [NSNumber]
                    array[idx] = NSNumber(value: array[idx].floatValue + scale * bias[c])
                }
            }
        }
    }
}
