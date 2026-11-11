// Usage: swift placement.swift <model.mlmodelc> [--predict N]
// Prints per-op preferred compute device from MLComputePlan (ANE/GPU/CPU), op-type breakdown
// of non-ANE ops, estimated cost share per device, and optionally times N predictions on
// .cpuAndNeuralEngine with peak RSS.
import CoreML
import Foundation

func rssMB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.resident_size_max) / 1_048_576 : -1
}

func devName(_ d: MLComputeDevice?) -> String {
    switch d {
    case .some(.neuralEngine): return "ANE"
    case .some(.gpu): return "GPU"
    case .some(.cpu): return "CPU"
    default: return "none"
    }
}

let args = CommandLine.arguments
let url = URL(fileURLWithPath: args[1])
let predictN = args.firstIndex(of: "--predict").map { Int(args[$0 + 1])! } ?? 0

let config = MLModelConfiguration()
switch ProcessInfo.processInfo.environment["UNITS"] ?? "ane" {
case "cpu": config.computeUnits = .cpuOnly
case "gpu": config.computeUnits = .cpuAndGPU
default: config.computeUnits = .cpuAndNeuralEngine
}

let sem = DispatchSemaphore(value: 0)
Task {
    defer { sem.signal() }
    do {
        let t0 = Date()
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
        let planSec = Date().timeIntervalSince(t0)
        guard case .program(let program) = plan.modelStructure,
              let main = program.functions["main"] else { print("not an ML program"); return }

        var countBy: [String: Int] = [:]
        var costBy: [String: Double] = [:]
        var nonANE: [String: [String: Int]] = [:]   // device -> opType -> count
        var supportedButNot: [String: Int] = [:]    // ops that list ANE as supported but weren't placed there

        func walk(_ block: MLModelStructure.Program.Block) {
            for op in block.operations {
                for b in op.blocks { walk(b) }
                if op.operatorName.hasPrefix("const") { continue }
                guard let usage = plan.deviceUsage(for: op) else { continue }
                let dev = devName(usage.preferred)
                countBy[dev, default: 0] += 1
                costBy[dev, default: 0] += plan.estimatedCost(of: op)?.weight ?? 0
                if dev != "ANE" {
                    nonANE[dev, default: [:]][op.operatorName, default: 0] += 1
                    if usage.supported.contains(where: { if case .neuralEngine = $0 { return true }; return false }) {
                        supportedButNot[op.operatorName, default: 0] += 1
                    }
                }
            }
        }
        walk(main.block)

        print("model: \(url.lastPathComponent)  (plan load \(String(format: "%.1f", planSec))s)")
        let total = countBy.values.reduce(0, +)
        for dev in ["ANE", "GPU", "CPU", "none"] where countBy[dev] != nil {
            print(String(format: "  %-4@ ops=%4d (%5.1f%%)  est.cost=%5.1f%%",
                         dev as NSString, countBy[dev]!, 100.0 * Double(countBy[dev]!) / Double(total),
                         100.0 * (costBy[dev] ?? 0)))
        }
        for (dev, ops) in nonANE.sorted(by: { $0.key < $1.key }) {
            let s = ops.sorted { $0.value > $1.value }.map { "\($0.key)×\($0.value)" }.joined(separator: ", ")
            print("  on \(dev): \(s)")
        }
        if !supportedButNot.isEmpty {
            print("  ANE-supported but placed elsewhere: \(supportedButNot.map { "\($0.key)×\($0.value)" }.joined(separator: ", "))")
        }

        if predictN > 0 {
            let tl = Date()
            let model = try MLModel(contentsOf: url, configuration: config)
            print(String(format: "  load(\(ProcessInfo.processInfo.environment["UNITS"] ?? "ane")) %.2fs  peakRSS %.0f MB", Date().timeIntervalSince(tl), rssMB()))
            var feats: [String: MLFeatureValue] = [:]
            for (name, desc) in model.modelDescription.inputDescriptionsByName {
                let shape = desc.multiArrayConstraint!.shape
                let arr = try MLMultiArray(shape: shape, dataType: .float32)
                let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
                for i in 0..<arr.count { p[i] = Float.random(in: -1...1) }
                feats[name] = MLFeatureValue(multiArray: arr)
            }
            let input = try MLDictionaryFeatureProvider(dictionary: feats)
            _ = try await model.prediction(from: input)   // warmup
            let tp = Date()
            for _ in 0..<predictN { _ = try await model.prediction(from: input) }
            let ms = Date().timeIntervalSince(tp) / Double(predictN) * 1000
            print(String(format: "  predict avg %.1f ms over %d  peakRSS %.0f MB", ms, predictN, rssMB()))
        }
    } catch {
        print("ERROR: \(error)")
    }
}
sem.wait()
