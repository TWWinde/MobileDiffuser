// FLUX.2 Klein (4B) on Core ML / ANE.
//
// Everything is a static-shape Core ML stage that is loaded, run and released in turn, so only one
// stage is resident at a time:
//   text      Qwen3-4B, first 27 layers as TextStage0..8 (3 layers each); the prompt embedding is the
//             concatenation of the hidden states after layers 9 / 18 / 27 -> [1, T, 7680]
//   denoiser  Embed, Double0..4, Single0..19, Final (one transformer block per stage)
//   VAE       KleinVAEDecoder (64x64x32 latent tile -> 512x512), 3x3 feather-blended tiles for 1024
//
// Resource folder (see tools/ane_probe/scripts/convert_flux2_klein_split.py, convert_qwen3_split.py,
// convert_flux2_vae.py): the .mlmodelc stages, flux2_klein.json, qwen3.json, vae_bn.json,
// rope_cos.bin / rope_sin.bin, qwen3_rope_cos.bin / qwen3_rope_sin.bin, embed_tokens.f16,
// tokenizer.json, tokenizer_config.json.

import Accelerate
import CoreGraphics
import CoreML
import Foundation
import Hub
import Tokenizers

@available(iOS 17.0, macOS 14.0, *)
public final class Flux2KleinPipeline {
    public enum KleinError: LocalizedError {
        case missingResource(String)
        case badResource(String)
        case nonFinite(String)
        case cancelled

        public var errorDescription: String? {
            switch self {
            case .missingResource(let n): return "Missing FLUX.2 Klein resource: \(n)"
            case .badResource(let n): return "Invalid FLUX.2 Klein resource: \(n)"
            case .nonFinite(let n): return "Non-finite values in \(n)"
            case .cancelled: return "Generation cancelled"
            }
        }
    }

    public enum Phase: Sendable {
        case encodingText
        case denoising(step: Int, total: Int)
        case decoding
    }

    public struct Timings: Sendable {
        public var text = 0.0, denoise = 0.0, decode = 0.0, load = 0.0
    }

    /// Per-resolution files (denoiser) and files shared by every resolution (text encoder, VAE, tokenizer).
    public static func denoiserResourceNames(doubleBlocks: Int = 5, singleBlocks: Int = 20) -> [String] {
        ["flux2_klein.json", "rope_cos.bin", "rope_sin.bin", "Embed.mlmodelc", "Final.mlmodelc"]
            + (0..<doubleBlocks).map { "Double\($0).mlmodelc" }
            + (0..<singleBlocks).map { "Single\($0).mlmodelc" }
    }

    public static func sharedResourceNames(textStages: Int = 9) -> [String] {
        ["qwen3.json", "vae_bn.json", "qwen3_rope_cos.bin", "qwen3_rope_sin.bin", "embed_tokens.f16",
         "tokenizer.json", "tokenizer_config.json", "KleinVAEDecoder.mlmodelc"]
            + (0..<textStages).map { "TextStage\($0).mlmodelc" }
    }

    /// Sibling folders searched (after the resolution folder itself) for the shared files, so the
    /// 3.5 GB text encoder is stored once.
    public static let sharedFolderNames = ["coreml_flux2_klein_common", "coreml_flux2_klein_1024"]

    static func locate(_ name: String, root: URL) -> URL? {
        let fm = FileManager.default
        let parent = root.deletingLastPathComponent()
        for dir in [root] + sharedFolderNames.map({ parent.appending(path: $0, directoryHint: .isDirectory) }) {
            let u = dir.appending(path: name)
            if fm.fileExists(atPath: u.path) { return u }
        }
        return nil
    }

    /// Names of required files that cannot be found for the resolution folder `root`.
    public static func missingResources(at root: URL) -> [String] {
        let fm = FileManager.default
        return denoiserResourceNames().filter { !fm.fileExists(atPath: root.appending(path: $0).path) }
            + sharedResourceNames().filter { locate($0, root: root) == nil }
    }

    let root: URL
    let configuration: MLModelConfiguration
    // denoiser
    let imageTokens: Int, textTokens: Int, doubleBlocks: Int, singleBlocks: Int
    let latentChannels = 128, textDim = 7680
    // text encoder
    let hidden: Int, textStages: Int, layersPerStage: Int, taps: [Int], vocab: Int, padToken: Int
    let template: String
    let tokenizer: Tokenizer
    let bnMean: [Float], bnStd: [Float]
    public private(set) var lastTimings = Timings()
    private var isCancelled: () -> Bool = { false }

    private func url(_ name: String) throws -> URL {
        guard let u = Self.locate(name, root: root) else { throw KleinError.missingResource(name) }
        return u
    }

    public init(resourcesAt root: URL, configuration: MLModelConfiguration) throws {
        self.root = root
        self.configuration = configuration
        func json(_ name: String) throws -> [String: Any] {
            guard let url = Self.locate(name, root: root) else { throw KleinError.missingResource(name) }
            guard let d = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
                throw KleinError.badResource(name)
            }
            return d
        }
        let k = try json("flux2_klein.json"), q = try json("qwen3.json"), v = try json("vae_bn.json")
        guard let n = k["image_tokens"] as? Int, let t = k["text_tokens"] as? Int,
              let nd = k["double_blocks"] as? Int, let ns = k["single_blocks"] as? Int else {
            throw KleinError.badResource("flux2_klein.json")
        }
        imageTokens = n; textTokens = t; doubleBlocks = nd; singleBlocks = ns
        guard let h = q["hidden"] as? Int, let st = q["stages"] as? Int, let lps = q["layers_per_stage"] as? Int,
              let tp = q["taps_after_layer"] as? [Int], let vc = q["vocab"] as? Int,
              let pad = q["pad_token_id"] as? Int, let tpl = q["template"] as? String,
              (q["text_tokens"] as? Int) == t else {
            throw KleinError.badResource("qwen3.json")
        }
        hidden = h; textStages = st; layersPerStage = lps; taps = tp; vocab = vc; padToken = pad; template = tpl
        guard let mean = v["mean"] as? [NSNumber], let std = v["std"] as? [NSNumber],
              mean.count == 128, std.count == 128 else { throw KleinError.badResource("vae_bn.json") }
        bnMean = mean.map { $0.floatValue }; bnStd = std.map { $0.floatValue }

        // Qwen2 tokenizer = byte-level BPE with added special tokens; swift-transformers has no class
        // entry for "Qwen2Tokenizer", so load it through the generic BPE tokenizer.
        var tc = try json("tokenizer_config.json")
        tc["tokenizer_class"] = "PreTrainedTokenizer"
        var td = try json("tokenizer.json")
        // Newer tokenizer.json files store merges as [a, b] pairs; this BPE implementation wants "a b".
        if var model = td["model"] as? [String: Any], let pairs = model["merges"] as? [[String]] {
            model["merges"] = pairs.map { $0.joined(separator: " ") }
            td["model"] = model
        }
        tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(tc), tokenizerData: Config(td))
    }

    // MARK: - Helpers

    private func loadFloats(_ name: String, shape: [Int]) throws -> MLMultiArray {
        guard let d = try? Data(contentsOf: try url(name), options: .alwaysMapped) else { throw KleinError.missingResource(name) }
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        guard d.count == a.count * 4 else { throw KleinError.badResource(name) }
        d.withUnsafeBytes { _ = memcpy(a.dataPointer, $0.baseAddress!, d.count) }
        return a
    }

    private func array(_ shape: [Int]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    }

    /// Float32 copy of a (possibly float16 / non-contiguous) model output.
    private func floats(_ a: MLMultiArray) -> [Float] {
        if a.dataType == .float32 {
            return a.withUnsafeBufferPointer(ofType: Float.self) { Array($0) }
        }
        return MLShapedArray<Float>(converting: a).scalars
    }

    /// A stage being loaded on a background queue while the previous one runs.
    private final class Pending {
        let name: String
        let group = DispatchGroup()
        var result: Result<MLModel, Error>?
        init(name: String) { self.name = name }
    }
    private var pending: Pending?
    private let prefetchQueue = DispatchQueue(label: "klein.stage-prefetch", qos: .userInitiated)
    /// Set false to load every stage synchronously (for A/B timing).
    public var prefetchStages = true

    private func prefetch(_ name: String) {
        guard prefetchStages, let u = Self.locate("\(name).mlmodelc", root: root) else { return }
        let p = Pending(name: name)
        p.group.enter()
        pending = p
        let configuration = self.configuration
        prefetchQueue.async {
            p.result = Result { try MLModel(contentsOf: u, configuration: configuration) }
            p.group.leave()
        }
    }

    /// The model for `name`: the prefetched one if it is the stage that was announced, else a fresh load.
    private func model(_ name: String) throws -> MLModel {
        let t0 = CFAbsoluteTimeGetCurrent()
        defer { lastTimings.load += CFAbsoluteTimeGetCurrent() - t0 }
        if let p = pending {
            pending = nil
            p.group.wait()                       // never leave a load running behind our back
            if p.name == name, let r = p.result { return try r.get() }
        }
        return try MLModel(contentsOf: try url("\(name).mlmodelc"), configuration: configuration)
    }

    private func dropPending() {
        if let p = pending { pending = nil; p.group.wait() }
    }

    /// Load one stage (or take the prefetched one), start loading `next`, run, release.
    private func run(_ name: String, next: String? = nil, _ feed: [String: Any]) throws -> MLFeatureProvider {
        if isCancelled() { dropPending(); throw KleinError.cancelled }
        let m = try model(name)
        if let next { prefetch(next) }
        return try m.prediction(from: MLDictionaryFeatureProvider(dictionary: feed))
    }

    private func out(_ r: MLFeatureProvider, _ key: String) throws -> MLMultiArray {
        guard let a = r.featureValue(for: key)?.multiArrayValue else { throw KleinError.badResource("output \(key)") }
        return a
    }

    // MARK: - Text

    /// Token ids for the chat-templated prompt, truncated/padded to `textTokens`; returns (ids, realCount).
    public func tokenize(_ prompt: String) -> (ids: [Int], count: Int) {
        // The template is encoded in two pieces. swift-transformers 0.1.8 locates added tokens with UTF-16
        // offsets, so a special token that follows an emoji in the same string is split incorrectly; the
        // special tokens of the prefix come before the prompt and the suffix is encoded on its own.
        let parts = template.components(separatedBy: "{}")
        var ids = tokenizer.encode(text: parts[0] + prompt)
        let suffix = parts.count > 1 ? tokenizer.encode(text: parts[1]) : []
        if ids.count + suffix.count > textTokens {
            ids = Array(ids.prefix(Swift.max(0, textTokens - suffix.count)))   // keep the assistant turn
        }
        ids += suffix
        if ids.count > textTokens { ids = Array(ids.prefix(textTokens)) }
        let n = ids.count
        ids += Array(repeating: padToken, count: textTokens - n)
        return (ids, n)
    }

    /// Prompt embedding [1, T, 7680].
    public func encodeText(_ prompt: String) throws -> MLMultiArray {
        let (ids, n) = tokenize(prompt)
        let T = textTokens
        // embedding lookup from the memory-mapped float16 table
        let table = try Data(contentsOf: try url("embed_tokens.f16"), options: .alwaysMapped)
        guard table.count == vocab * hidden * 2 else { throw KleinError.badResource("embed_tokens.f16") }
        var x = try array([1, T, hidden])
        table.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: UInt16.self)
            let dst = x.dataPointer.assumingMemoryBound(to: Float.self)
            for (t, id) in ids.enumerated() {
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src.baseAddress! + id * hidden),
                                      height: 1, width: vImagePixelCount(hidden), rowBytes: hidden * 2)
                var d = vImage_Buffer(data: dst + t * hidden, height: 1, width: vImagePixelCount(hidden),
                                      rowBytes: hidden * 4)
                vImageConvert_Planar16FtoPlanarF(&s, &d, 0)
            }
        }
        // additive mask: causal, padded keys masked
        let mask = try array([1, 1, T, T])
        let mp = mask.dataPointer.assumingMemoryBound(to: Float.self)
        for r in 0..<T { for c in 0..<T { mp[r * T + c] = (c > r || c >= n) ? -1e4 : 0 } }
        let cos = try loadFloats("qwen3_rope_cos.bin", shape: [T, 128])
        let sin = try loadFloats("qwen3_rope_sin.bin", shape: [T, 128])

        let text = try array([1, T, textDim])
        let tp = text.dataPointer.assumingMemoryBound(to: Float.self)
        var tapIndex = 0
        for s in 0..<textStages {
            x = try out(try run("TextStage\(s)", next: s + 1 < textStages ? "TextStage\(s + 1)" : "Embed",
                                ["hidden": x, "mask": mask, "rope_cos": cos, "rope_sin": sin]), "hidden_out")
            if taps.contains((s + 1) * layersPerStage) {
                let h = floats(x)
                guard h.allSatisfy({ $0.isFinite }) else { throw KleinError.nonFinite("text encoder stage \(s)") }
                for t in 0..<T {
                    for d in 0..<hidden { tp[t * textDim + tapIndex * hidden + d] = h[t * hidden + d] }
                }
                tapIndex += 1
            }
        }
        guard tapIndex * hidden == textDim else { throw KleinError.badResource("qwen3.json taps") }
        return text
    }

    // MARK: - Schedule

    /// FLUX.2 flow-matching sigmas (empirical mu, exponential time shift), `steps + 1` values ending in 0.
    public static func sigmas(imageTokens: Int, steps: Int) -> [Double] {
        let a1 = 8.73809524e-05, b1 = 1.89833333, a2 = 0.00016927, b2 = 0.45666666
        let n = Double(imageTokens)
        let mu: Double
        if imageTokens > 4300 {
            mu = a2 * n + b2
        } else {
            let m200 = a2 * n + b2, m10 = a1 * n + b1
            let a = (m200 - m10) / 190.0
            mu = a * Double(steps) + (m200 - 200.0 * a)
        }
        var out: [Double] = []
        for i in 0..<steps {
            let s = steps == 1 ? 1.0 : 1.0 + Double(i) * (1.0 / Double(steps) - 1.0) / Double(steps - 1)
            out.append(exp(mu) / (exp(mu) + (1.0 / s - 1.0)))
        }
        return out + [0]
    }

    // MARK: - Generate

    /// Euler steps over the consecutive sigmas `sig`, updating `latent` [1, N, 128] in place.
    /// Returns the clean-image estimate (x0 = x - sigma * v) of the last step.
    private func denoise(latent: MLMultiArray, text: MLMultiArray, sig: [Double], afterLast: String?,
                         stepBase: Int, totalSteps: Int, progress: (Phase) -> Bool) throws -> [Float] {
        let N = imageTokens, T = textTokens, C = latentChannels
        let lp = latent.dataPointer.assumingMemoryBound(to: Float.self)
        let cos = try loadFloats("rope_cos.bin", shape: [T + N, 128])
        let sin = try loadFloats("rope_sin.bin", shape: [T + N, 128])
        var x0 = [Float](repeating: 0, count: N * C)
        let count = sig.count - 1
        for s in 0..<count {
            guard progress(.denoising(step: stepBase + s + 1, total: totalSteps)) else { throw KleinError.cancelled }
            let ts = try array([1])
            ts[0] = NSNumber(value: Float(sig[s]))
            let e = try run("Embed", next: "Double0", ["latent": latent, "text": text, "timestep": ts])
            var img = try out(e, "img"), txt = try out(e, "txt")
            let modImg = try out(e, "mod_img"), modTxt = try out(e, "mod_txt")
            for i in 0..<doubleBlocks {
                let d = try run("Double\(i)", next: i + 1 < doubleBlocks ? "Double\(i + 1)" : "Single0",
                                ["img": img, "txt": txt, "mod_img": modImg, "mod_txt": modTxt,
                                 "rope_cos": cos, "rope_sin": sin])
                img = try out(d, "img_out"); txt = try out(d, "txt_out")
            }
            var hiddenState = MLMultiArray(concatenating: [txt, img], axis: 1, dataType: .float32)
            let modSingle = try out(e, "mod_single")
            for i in 0..<singleBlocks {
                hiddenState = try out(try run("Single\(i)", next: i + 1 < singleBlocks ? "Single\(i + 1)" : "Final",
                                              ["hidden": hiddenState, "mod": modSingle,
                                               "rope_cos": cos, "rope_sin": sin]), "hidden_out")
            }
            let noise = floats(try out(try run("Final", next: s + 1 < count ? "Embed" : afterLast,
                                               ["hidden": hiddenState, "temb": try out(e, "temb")]), "noise"))
            guard noise.count == N * C, noise.allSatisfy({ $0.isFinite }) else {
                throw KleinError.nonFinite("denoiser step \(stepBase + s + 1)")
            }
            noise.withUnsafeBufferPointer { np in
                if s == count - 1 {
                    var m = -Float(sig[s])
                    vDSP_vsma(np.baseAddress!, 1, &m, lp, 1, &x0, 1, vDSP_Length(N * C))
                }
                var d = Float(sig[s + 1] - sig[s])
                vDSP_vsma(np.baseAddress!, 1, &d, lp, 1, lp, 1, vDSP_Length(N * C))
            }
        }
        return x0
    }

    /// Bicubic x2 upsampling (PyTorch `interpolate(mode: bicubic, align_corners: false)`) of a packed,
    /// normalised latent [side*side, 128], done in the unpatchified 32-channel VAE space.
    func upsample2x(packed lat: [Float], side: Int) -> [Float] {
        let h = side * 2, H = h * 2                     // unpatchified size before / after
        var z = [Float](repeating: 0, count: 32 * h * h)
        for y in 0..<side {
            for x in 0..<side {
                let base = (y * side + x) * 128
                for ch in 0..<128 {
                    let c = ch / 4, dy = (ch % 4) / 2, dx = ch % 2
                    z[c * h * h + (2 * y + dy) * h + (2 * x + dx)] = lat[base + ch] * bnStd[ch] + bnMean[ch]
                }
            }
        }
        func weights(_ t: Float) -> [Float] {           // cubic convolution, A = -0.75
            let A: Float = -0.75
            func near(_ x: Float) -> Float { ((A + 2) * x - (A + 3)) * x * x + 1 }
            func far(_ x: Float) -> Float { ((A * x - 5 * A) * x + 8 * A) * x - 4 * A }
            return [far(t + 1), near(t), near(1 - t), far(2 - t)]
        }
        // taps for every output coordinate (same along both axes)
        var idx = [[Int]](), wts = [[Float]]()
        for o in 0..<H {
            let src = (Float(o) + 0.5) / 2 - 0.5
            let f = Int(src.rounded(.down))
            idx.append((-1...2).map { Swift.min(Swift.max(f + $0, 0), h - 1) })
            wts.append(weights(src - Float(f)))
        }
        var up = [Float](repeating: 0, count: 32 * H * H)
        var tmp = [Float](repeating: 0, count: h * H)   // horizontally resampled plane [h rows, H cols]
        for c in 0..<32 {
            for y in 0..<h {
                for X in 0..<H {
                    var v: Float = 0
                    for k in 0..<4 { v += wts[X][k] * z[c * h * h + y * h + idx[X][k]] }
                    tmp[y * H + X] = v
                }
            }
            for Y in 0..<H {
                for X in 0..<H {
                    var v: Float = 0
                    for k in 0..<4 { v += wts[Y][k] * tmp[idx[Y][k] * H + X] }
                    up[c * H * H + Y * H + X] = v
                }
            }
        }
        let S = side * 2
        var out = [Float](repeating: 0, count: S * S * 128)
        for y in 0..<S {
            for x in 0..<S {
                let base = (y * S + x) * 128
                for ch in 0..<128 {
                    let c = ch / 4, dy = (ch % 4) / 2, dx = ch % 2
                    out[base + ch] = (up[c * H * H + (2 * y + dy) * H + (2 * x + dx)] - bnMean[ch]) / bnStd[ch]
                }
            }
        }
        return out
    }

    /// - Parameter initialLatent: optional packed noise [N * 128] (overrides `seed`; used for parity checks).
    /// - Parameter isCancelled: polled before every stage; returning true aborts with `KleinError.cancelled`.
    /// - Parameter lowRes: a pipeline for the half-resolution stages. When given, the first `lowSteps` steps
    ///   of the schedule run there, the clean-image estimate is upsampled x2 and re-noised, and only the last
    ///   `highSteps` steps run at this pipeline's resolution ("low-res start, high-res finish").
    public func generateImage(prompt: String, seed: UInt32, steps: Int = 4, initialLatent: [Float]? = nil,
                              lowRes: Flux2KleinPipeline? = nil, lowSteps: Int = 3, highSteps: Int = 1,
                              isCancelled: @escaping () -> Bool = { false },
                              progress: (Phase) -> Bool = { _ in true }) throws -> CGImage {
        lastTimings = Timings()
        self.isCancelled = isCancelled
        lowRes?.isCancelled = isCancelled
        lowRes?.prefetchStages = prefetchStages
        lowRes?.lastTimings = Timings()
        defer {
            self.isCancelled = { false }; dropPending()
            lowRes?.isCancelled = { false }; lowRes?.dropPending()
        }
        let N = imageTokens, C = latentChannels
        guard progress(.encodingText) else { throw KleinError.cancelled }
        var t0 = CFAbsoluteTimeGetCurrent()
        let text = try encodeText(prompt)
        lastTimings.text = CFAbsoluteTimeGetCurrent() - t0
        dropPending()                                    // encodeText announces "Embed" of this resolution

        var rng = TorchRandomSource(seed: seed)
        let latent = try array([1, N, C])
        let lp = latent.dataPointer.assumingMemoryBound(to: Float.self)
        t0 = CFAbsoluteTimeGetCurrent()
        if let lo = lowRes {
            guard lo.imageTokens * 4 == N, lo.textTokens == textTokens, lowSteps >= 1, highSteps >= 1,
                  lowSteps <= steps, highSteps <= steps else { throw KleinError.badResource("lowRes configuration") }
            let n = lo.imageTokens
            let loLatent = try array([1, n, C])
            let lop = loLatent.dataPointer.assumingMemoryBound(to: Float.self)
            let noise = rng.normalArray(count: n * C)
            for i in 0..<(n * C) { lop[i] = Float(noise[i]) }
            let sigLo = Array(Self.sigmas(imageTokens: n, steps: steps).prefix(lowSteps + 1))
            let sigHi = Array(Self.sigmas(imageTokens: N, steps: steps).suffix(highSteps + 1))
            let total = lowSteps + highSteps
            let x0 = try lo.denoise(latent: loLatent, text: text, sig: sigLo, afterLast: nil,
                                    stepBase: 0, totalSteps: total, progress: progress)
            let up = upsample2x(packed: x0, side: Int(Double(n).squareRoot()))
            let eps = rng.normalArray(count: N * C)
            let sr = Float(sigHi[0])
            for i in 0..<(N * C) { lp[i] = (1 - sr) * up[i] + sr * Float(eps[i]) }
            _ = try denoise(latent: latent, text: text, sig: sigHi, afterLast: "KleinVAEDecoder",
                            stepBase: lowSteps, totalSteps: total, progress: progress)
            lastTimings.load += lo.lastTimings.load
        } else {
            if let initialLatent {
                guard initialLatent.count == N * C else { throw KleinError.badResource("initialLatent") }
                for i in 0..<(N * C) { lp[i] = initialLatent[i] }
            } else {
                let noise = rng.normalArray(count: N * C)
                for i in 0..<(N * C) { lp[i] = Float(noise[i]) }
            }
            _ = try denoise(latent: latent, text: text, sig: Self.sigmas(imageTokens: N, steps: steps),
                            afterLast: "KleinVAEDecoder", stepBase: 0, totalSteps: steps, progress: progress)
        }
        lastTimings.denoise = CFAbsoluteTimeGetCurrent() - t0

        guard progress(.decoding) else { throw KleinError.cancelled }
        t0 = CFAbsoluteTimeGetCurrent()
        let image = try decode(packedLatent: Array(UnsafeBufferPointer(start: lp, count: N * C)))
        lastTimings.decode = CFAbsoluteTimeGetCurrent() - t0
        return image
    }

    // MARK: - VAE

    /// packed [N, 128] -> batch-norm denormalise -> unpatchify [32, 2s, 2s] -> feather-blended tiled decode.
    public func decode(packedLatent lat: [Float]) throws -> CGImage {
        let side = Int(Double(imageTokens).squareRoot())            // 64 for 1024 px
        let H = side * 2, W = side * 2, tile = 64, scale = 8
        guard side * side == imageTokens, H >= tile else { throw KleinError.badResource("image_tokens") }
        // z[c, 2y+dy, 2x+dx] = lat[y*side+x, c*4 + dy*2 + dx] * std + mean
        var z = [Float](repeating: 0, count: 32 * H * W)
        for y in 0..<side {
            for x in 0..<side {
                let base = (y * side + x) * 128
                for ch in 0..<128 {
                    let c = ch / 4, dy = (ch % 4) / 2, dx = ch % 2
                    z[c * H * W + (2 * y + dy) * W + (2 * x + dx)] = lat[base + ch] * bnStd[ch] + bnMean[ch]
                }
            }
        }
        let stride = tile / 2
        var starts: [Int] = Array(Swift.stride(from: 0, through: H - tile, by: stride))
        if starts.last != H - tile { starts.append(H - tile) }
        let outH = H * scale, outW = W * scale, tp = tile * scale
        var acc = [Float](repeating: 0, count: 3 * outH * outW)
        var wsum = [Float](repeating: 0, count: outH * outW)
        let ramp: [Float] = (0..<tp).map { Swift.min(Float($0) + 0.5, Float(tp - $0) - 0.5) }

        let model = try self.model("KleinVAEDecoder")
        let input = try array([1, 32, tile, tile])
        let ip = input.dataPointer.assumingMemoryBound(to: Float.self)
        var rowW = [Float](repeating: 0, count: tp)
        for ty in starts {
            for tx in starts {
                if isCancelled() { throw KleinError.cancelled }
                z.withUnsafeBufferPointer { zp in
                    for c in 0..<32 {
                        for y in 0..<tile {
                            memcpy(ip + (c * tile + y) * tile, zp.baseAddress! + c * H * W + (ty + y) * W + tx,
                                   tile * MemoryLayout<Float>.size)
                        }
                    }
                }
                let r = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["latent": input]))
                let px = floats(try out(r, "image"))
                guard px.count == 3 * tp * tp else { throw KleinError.badResource("VAE output shape") }
                let oy = ty * scale, ox = tx * scale
                px.withUnsafeBufferPointer { pp in
                    for y in 0..<tp {
                        var wy = ramp[y]
                        vDSP_vsmul(ramp, 1, &wy, &rowW, 1, vDSP_Length(tp))            // w(y, x) = ramp[y] * ramp[x]
                        let o = (oy + y) * outW + ox
                        wsum.withUnsafeMutableBufferPointer {
                            vDSP_vadd($0.baseAddress! + o, 1, rowW, 1, $0.baseAddress! + o, 1, vDSP_Length(tp))
                        }
                        acc.withUnsafeMutableBufferPointer { ap in
                            for c in 0..<3 {
                                let dst = ap.baseAddress! + c * outH * outW + o
                                vDSP_vma(pp.baseAddress! + (c * tp + y) * tp, 1, rowW, 1, dst, 1, dst, 1, vDSP_Length(tp))
                            }
                        }
                    }
                }
            }
        }
        // acc / wsum -> [0, 255] -> interleaved RGB8
        let plane = outH * outW
        var rgb = [UInt8](repeating: 0, count: plane * 3)
        var tmp = [Float](repeating: 0, count: plane)
        var one: Float = 1, half: Float = 127.5, lo: Float = 0, hi: Float = 255
        for c in 0..<3 {
            acc.withUnsafeBufferPointer { ap in
                vDSP_vdiv(wsum, 1, ap.baseAddress! + c * plane, 1, &tmp, 1, vDSP_Length(plane))   // acc / wsum
            }
            guard tmp[0].isFinite, tmp[plane / 2].isFinite, tmp[plane - 1].isFinite else {
                throw KleinError.nonFinite("VAE output")
            }
            vDSP_vsadd(tmp, 1, &one, &tmp, 1, vDSP_Length(plane))
            vDSP_vsmul(tmp, 1, &half, &tmp, 1, vDSP_Length(plane))
            vDSP_vclip(tmp, 1, &lo, &hi, &tmp, 1, vDSP_Length(plane))
            rgb.withUnsafeMutableBufferPointer { rp in
                vDSP_vfixru8(tmp, 1, rp.baseAddress! + c, 3, vDSP_Length(plane))
            }
        }
        guard let provider = CGDataProvider(data: Data(rgb) as CFData),
              let image = CGImage(width: outW, height: outH, bitsPerComponent: 8, bitsPerPixel: 24,
                                  bytesPerRow: outW * 3, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) else {
            throw KleinError.badResource("image creation")
        }
        return image
    }
}
