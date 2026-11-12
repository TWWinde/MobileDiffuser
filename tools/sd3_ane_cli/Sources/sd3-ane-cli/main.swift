import CoreGraphics
import CoreML
import Foundation
import ImageIO
import StableDiffusion
import UniformTypeIdentifiers

// Usage:
//   sd3-ane-cli --resources <coremlsd3_4step_1024> --prompt "..." --steps 4 --seed 42 \
//               --size 1024 --out out.png [--units ane|gpu|cpu]
var opts: [String: String] = [:]
var i = 1
let argv = CommandLine.arguments
while i + 1 < argv.count, argv[i].hasPrefix("--") {
    opts[String(argv[i].dropFirst(2))] = argv[i + 1]
    i += 2
}
guard let resources = opts["resources"], let out = opts["out"] else {
    print("usage: sd3-ane-cli --resources DIR --out FILE.png [--prompt P] [--steps 4] [--seed 42] [--size 1024] [--units ane|gpu|cpu] [--reseed 1.0]")
    exit(2)
}
let size = Float32(opts["size"] ?? "1024")!

let mlConfig = MLModelConfiguration()
switch opts["units"] ?? "ane" {
case "cpu": mlConfig.computeUnits = .cpuOnly
case "gpu": mlConfig.computeUnits = .cpuAndGPU
default: mlConfig.computeUnits = .cpuAndNeuralEngine
}

func savePNG(_ image: CGImage, _ path: String) {
    let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
    print("saved \(path) \(image.width)x\(image.height)")
}

// FLUX.2 Klein:  sd3-ane-cli --klein 1 --resources <coreml_flux2_klein_1024> --prompt "..." --out out.png
//                [--latent packed_noise.bin] [--tokens 1 (print token ids and exit)]
if opts["klein"] != nil {
    let klein = try Flux2KleinPipeline(resourcesAt: URL(fileURLWithPath: resources), configuration: mlConfig)
    let prompt = opts["prompt"] ?? "A photo of a red fox in a snowy forest, highly detailed"
    if opts["tokens"] != nil {
        let (ids, n) = klein.tokenize(prompt)
        print("tokens \(n): \(ids.prefix(n).map(String.init).joined(separator: " "))")
        exit(0)
    }
    var initial: [Float]? = nil
    if let f = opts["latent"] {
        let d = try Data(contentsOf: URL(fileURLWithPath: f))
        initial = d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    let tk = CFAbsoluteTimeGetCurrent()
    let image = try klein.generateImage(prompt: prompt, seed: UInt32(opts["seed"] ?? "42")!,
                                        steps: Int(opts["steps"] ?? "4")!, initialLatent: initial) { phase in
        print("phase \(phase)")
        return true
    }
    let tm = klein.lastTimings
    print(String(format: "total %.1fs  text %.1fs  denoise %.1fs  decode %.1fs  (model load %.1fs)",
                 CFAbsoluteTimeGetCurrent() - tk, tm.text, tm.denoise, tm.decode, tm.load))
    savePNG(image, out)
    exit(0)
}

let t0 = CFAbsoluteTimeGetCurrent()
let pipeline = try StableDiffusion3Pipeline(
    resourcesAt: URL(fileURLWithPath: resources),
    configuration: mlConfig,
    reduceMemory: true
)

var config = StableDiffusionPipeline.Configuration(prompt: opts["prompt"] ?? "A photo of a red fox in a snowy forest, highly detailed")
config.stepCount = Int(opts["steps"] ?? "4")!
config.guidanceScale = 1.0
config.schedulerTimestepShift = 3.0
config.seed = UInt32(opts["seed"] ?? "42")!
config.encoderScaleFactor = 1.5305
config.decoderScaleFactor = 1.5305
config.decoderShiftFactor = 0.0609
config.originalSize = size
config.targetSize = size
config.disableSafety = true
config.reseedScale = Float(opts["reseed"] ?? "0")!

let images = try pipeline.generateImages(configuration: config) { progress in
    print("step \(progress.step + 1)/\(progress.stepCount)")
    return true
}
print(String(format: "total %.1fs", CFAbsoluteTimeGetCurrent() - t0))
guard let image = images.first ?? nil else { print("no image"); exit(1) }
let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: out) as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(dest, image, nil)
CGImageDestinationFinalize(dest)
print("saved \(out) \(image.width)x\(image.height)")
