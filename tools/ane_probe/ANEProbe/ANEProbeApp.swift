// ANEProbe — load one bundled Core ML single-block model on device with .cpuAndNeuralEngine,
// report MLComputePlan placement, load time, prediction latency and memory footprint.
//
// Autorun (for `devicectl device process launch --console`):
//   ANEProbe --model r1024_qchunk16 --predict 5      -> prints [PROBE] lines, then exits
// Without arguments it shows a list; tap a model to run it (results also printed to console).
import CoreML
import Darwin
import SwiftUI

enum Mem {
    static func footprintMB() -> (now: Double, peak: Double) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return (-1, -1) }
        return (Double(info.phys_footprint) / 1_048_576, Double(info.ledger_phys_footprint_peak) / 1_048_576)
    }
    static func availableMB() -> Double { Double(os_proc_available_memory()) / 1_048_576 }
}

func log(_ s: String) {
    print("[PROBE] \(s)")
    fflush(stdout)
}

func memLine(_ tag: String) -> String {
    let f = Mem.footprintMB()
    return String(format: "%@ footprint=%.0fMB peak=%.0fMB avail=%.0fMB", tag, f.now, f.peak, Mem.availableMB())
}

func runProbe(model name: String, predictN: Int) async -> String {
    var out: [String] = []
    func emit(_ s: String) { log(s); out.append(s) }
    guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc", subdirectory: "Models") else {
        emit("model \(name) not found in bundle"); return out.joined(separator: "\n")
    }
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    let os = ProcessInfo.processInfo.operatingSystemVersionString
    emit("model=\(name) os=\(os) physMem=\(ProcessInfo.processInfo.physicalMemory / 1_048_576)MB")
    emit(memLine("start"))
    do {
        // 1) Placement
        let tp = Date()
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
        var counts: [String: Int] = [:]
        var nonANE: [String: Int] = [:]
        if case .program(let program) = plan.modelStructure, let main = program.functions["main"] {
            func walk(_ b: MLModelStructure.Program.Block) {
                for op in b.operations {
                    op.blocks.forEach(walk)
                    if op.operatorName.hasPrefix("const") { continue }
                    guard let u = plan.deviceUsage(for: op) else { continue }
                    let d: String
                    switch u.preferred {
                    case .neuralEngine: d = "ANE"
                    case .gpu: d = "GPU"
                    case .cpu: d = "CPU"
                    @unknown default: d = "?"
                    }
                    counts[d, default: 0] += 1
                    if d != "ANE" { nonANE["\(d):\(op.operatorName)", default: 0] += 1 }
                }
            }
            walk(main.block)
        }
        emit(String(format: "plan %.2fs placement=%@ nonANE=%@", Date().timeIntervalSince(tp),
                    counts.description, nonANE.description))
        emit(memLine("after plan"))

        // 2) Load (first load on device triggers ANE compile; second hits the cache)
        let tl = Date()
        let model = try MLModel(contentsOf: url, configuration: config)
        emit(String(format: "load %.2fs", Date().timeIntervalSince(tl)))
        emit(memLine("after load"))

        // 3) Predict
        var feats: [String: MLFeatureValue] = [:]
        for (n, desc) in model.modelDescription.inputDescriptionsByName {
            let arr = try MLMultiArray(shape: desc.multiArrayConstraint!.shape, dataType: .float32)
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<arr.count { p[i] = Float.random(in: -1...1) }
            feats[n] = MLFeatureValue(multiArray: arr)
        }
        let input = try MLDictionaryFeatureProvider(dictionary: feats)
        let tw = Date()
        _ = try await model.prediction(from: input)
        emit(String(format: "warmup predict %.1fms", Date().timeIntervalSince(tw) * 1000))
        emit(memLine("after warmup"))
        var times: [Double] = []
        for _ in 0..<predictN {
            let t = Date()
            _ = try await model.prediction(from: input)
            times.append(Date().timeIntervalSince(t) * 1000)
        }
        if !times.isEmpty {
            emit(String(format: "predict avg=%.1fms min=%.1fms max=%.1fms n=%d",
                        times.reduce(0, +) / Double(times.count), times.min()!, times.max()!, times.count))
        }
        emit(memLine("end"))
        emit("RESULT OK \(name)")
    } catch {
        emit("ERROR \(name): \(error)")
        emit(memLine("at error"))
    }
    return out.joined(separator: "\n")
}

/// Run a full split-MMDiT stage chain (Models/<chain>/MultiModalDiffusionTransformerStage0..N.mlmodelc)
/// for `steps` denoising steps, loading and unloading each stage per step like the MCL app's
/// reduceMemory path. Inputs are synthetic (random latent/text, fixed modulation).
func runChain(_ chain: String, steps: Int) async -> String {
    var out: [String] = []
    func emit(_ s: String) { log(s); out.append(s) }
    guard let dir = Bundle.main.url(forResource: chain, withExtension: nil, subdirectory: "Models") else {
        emit("chain \(chain) not found"); return out.joined(separator: "\n")
    }
    var urls: [URL] = []
    var i = 0
    while true {
        let u = dir.appending(path: "MultiModalDiffusionTransformerStage\(i).mlmodelc")
        guard FileManager.default.fileExists(atPath: u.path) else { break }
        urls.append(u); i += 1
    }
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    emit("chain=\(chain) stages=\(urls.count) steps=\(steps)")
    emit(memLine("start"))
    do {
        func randomArray(_ shape: [NSNumber]) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: shape, dataType: .float32)
            let p = a.dataPointer.assumingMemoryBound(to: Float.self)
            for k in 0..<a.count { p[k] = Float.random(in: -1...1) }
            return a
        }
        // Discover shapes from Stage0
        let s0 = try MLModel(contentsOf: urls[0], configuration: config)
        let latentShape = s0.modelDescription.inputDescriptionsByName["latent_image_embeddings"]!.multiArrayConstraint!.shape
        let latent = try randomArray(latentShape)
        let text = try randomArray([1, 4096, 1, 154])
        let modulation = try randomArray([1, 1536])

        // First pass: compile/load every stage once (cold-start cost)
        let tc = Date()
        let keep = ProcessInfo.processInfo.arguments.contains("--keep")
        var resident: [MLModel] = []
        for (k, u) in urls.enumerated() {
            let t = Date()
            let m = try MLModel(contentsOf: u, configuration: config)
            if keep { resident.append(m) }
            emit(String(format: "first-load stage%d %.2fs", k, Date().timeIntervalSince(t)))
        }
        emit("mode=\(keep ? "keep-resident" : "reload-per-step")")
        emit(String(format: "first-load total %.1fs", Date().timeIntervalSince(tc)))
        emit(memLine("after first-load"))

        var stepTimes: [Double] = []
        for step in 0..<steps {
            let ts = Date()
            var loadSum = 0.0, predSum = 0.0
            var feats: [String: Any] = [
                "latent_image_embeddings": latent,
                "token_level_text_embeddings": text,
                "modulation_inputs": modulation,
            ]
            for (k, u) in urls.enumerated() {
                let tl = Date()
                let m = keep ? resident[k] : try MLModel(contentsOf: u, configuration: config)   // cached plan
                loadSum += Date().timeIntervalSince(tl)
                let tp = Date()
                let accepted = Set(m.modelDescription.inputDescriptionsByName.keys)
                let r = try await m.prediction(from: MLDictionaryFeatureProvider(dictionary: feats.filter { accepted.contains($0.key) }))
                predSum += Date().timeIntervalSince(tp)
                if let x = r.featureValue(for: "latent_image_embeddings_out")?.multiArrayValue {
                    feats["latent_image_embeddings"] = x
                    feats["token_level_text_embeddings"] = r.featureValue(for: "token_level_text_embeddings_out")!.multiArrayValue!
                }
            }
            let dt = Date().timeIntervalSince(ts)
            stepTimes.append(dt)
            emit(String(format: "step %d/%d total=%.2fs load=%.2fs predict=%.2fs", step + 1, steps, dt, loadSum, predSum))
            emit(memLine("after step \(step + 1)"))
        }
        emit(String(format: "MMDiT %d steps total=%.1fs", steps, stepTimes.reduce(0, +)))
        emit("RESULT OK \(chain)")
    } catch {
        emit("ERROR \(chain): \(error)")
        emit(memLine("at error"))
    }
    return out.joined(separator: "\n")
}

@main
struct ANEProbeApp: App {
    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "--chain"), i + 1 < args.count {
            let name = args[i + 1]
            let n = args.firstIndex(of: "--steps").flatMap { Int(args[$0 + 1]) } ?? 4
            Task.detached {
                _ = await runChain(name, steps: n)
                log("DONE")
                exit(0)
            }
        } else if let i = args.firstIndex(of: "--model"), i + 1 < args.count {
            let name = args[i + 1]
            let n = args.firstIndex(of: "--predict").flatMap { Int(args[$0 + 1]) } ?? 5
            Task.detached {
                _ = await runProbe(model: name, predictN: n)
                log("DONE")
                exit(0)
            }
        }
    }

    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @State private var output = "Tap a model. Run one per app launch for clean memory numbers."
    @State private var running = false

    var models: [String] {
        let dir = Bundle.main.resourceURL!.appending(path: "Models")
        return ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
            .filter { $0.hasSuffix(".mlmodelc") }
            .map { String($0.dropLast(".mlmodelc".count)) }
            .sorted()
    }

    var body: some View {
        NavigationStack {
            List {
                Section("Models") {
                    ForEach(models, id: \.self) { m in
                        Button(m) {
                            running = true
                            output = "running \(m)…"
                            Task {
                                output = await runProbe(model: m, predictN: 5)
                                running = false
                            }
                        }
                        .disabled(running)
                    }
                }
                Section("Output") {
                    Text(output).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                }
            }
            .navigationTitle("ANE Probe")
        }
    }
}
