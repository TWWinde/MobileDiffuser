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

import CoreML
import Foundation

@available(iOS 16.2, macOS 13.1, *)
public struct ReSeedBiases {
    /// Per-block image-stream bias (count = number of transformer blocks).
    public let image: [[Float]]
    /// Per-block text-stream bias; nil for the context-pre-only last block.
    public let context: [[Float]?]

    public static let resourceName = "ReSeedBiases.json"

    private struct Payload: Decodable {
        let image: [[Float]]
        let context: [[Float]?]
    }

    public init(contentsOf url: URL) throws {
        let payload = try JSONDecoder().decode(Payload.self, from: Data(contentsOf: url))
        self.image = payload.image
        self.context = payload.context
    }

    /// Adds `scale * bias` to every token of a [1, tokens, channels] Core ML array in place.
    static func add(_ bias: [Float], scale: Float, to array: MLMultiArray) {
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
