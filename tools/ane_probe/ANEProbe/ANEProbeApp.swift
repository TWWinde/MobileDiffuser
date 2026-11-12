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

/// FLUX.2 Klein split chain (Models/<chain>/{Embed,Double0..,Single0..,Final}.mlmodelc) with real inputs:
/// latent.bin [1,N,128], text.bin [1,T,7680], rope_cos.bin / rope_sin.bin [T+N,128] (float32), klein.json
/// {image_tokens,text_tokens,double_blocks,single_blocks,sigmas}. One stage resident at a time. Runs the
/// Euler flow-matching loop and writes the final latent to Documents/klein_latent.bin.
func runKlein(_ chain: String) async -> String {
    var out: [String] = []
    func emit(_ s: String) { log(s); out.append(s) }
    guard let dir = Bundle.main.url(forResource: chain, withExtension: nil, subdirectory: "Models") else {
        emit("klein chain \(chain) not found"); return out.joined(separator: "\n")
    }
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    do {
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appending(path: "klein.json"))) as! [String: Any]
        let n = meta["image_tokens"] as! Int, t = meta["text_tokens"] as! Int
        let nd = meta["double_blocks"] as! Int, ns = meta["single_blocks"] as! Int
        let sigmas = (meta["sigmas"] as! [NSNumber]).map { $0.doubleValue }
        func load(_ file: String, _ shape: [Int]) throws -> MLMultiArray {
            let d = try Data(contentsOf: dir.appending(path: file))
            let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
            precondition(d.count == a.count * 4, "\(file) size mismatch")
            d.withUnsafeBytes { memcpy(a.dataPointer, $0.baseAddress!, d.count) }
            return a
        }
        let latent = try load("latent.bin", [1, n, 128])
        let text = try load("text.bin", [1, t, 7680])
        let cos = try load("rope_cos.bin", [t + n, 128]), sin = try load("rope_sin.bin", [t + n, 128])
        emit("klein=\(chain) image_tokens=\(n) text_tokens=\(t) double=\(nd) single=\(ns) steps=\(sigmas.count - 1)")
        emit(memLine("start"))
        var loadSum = 0.0, predSum = 0.0, firstLoad = true
        var perKind: [String: [Double]] = [:]
        func run(_ name: String, _ feed: [String: Any]) async throws -> MLFeatureProvider {
            let tl = Date()
            let m = try MLModel(contentsOf: dir.appending(path: "\(name).mlmodelc"), configuration: config)
            let dl = Date().timeIntervalSince(tl)
            loadSum += dl
            if firstLoad { emit(String(format: "first-load %@ %.2fs", name, dl)) }
            let tp = Date()
            let r = try await m.prediction(from: MLDictionaryFeatureProvider(dictionary: feed))
            let dp = Date().timeIntervalSince(tp)
            predSum += dp
            perKind[String(name.prefix(while: { !$0.isNumber })), default: []].append(dp * 1000)
            return r
        }
        func arr(_ r: MLFeatureProvider, _ k: String) -> MLMultiArray { r.featureValue(for: k)!.multiArrayValue! }
        let lp = latent.dataPointer.assumingMemoryBound(to: Float.self)
        var stepTimes: [Double] = []
        for s in 0..<(sigmas.count - 1) {
            let ts = Date()
            loadSum = 0; predSum = 0; perKind = [:]
            let tsig = try MLMultiArray(shape: [1], dataType: .float32)
            tsig[0] = NSNumber(value: Float(sigmas[s]))
            let e = try await run("Embed", ["latent": latent, "text": text, "timestep": tsig])
            var img = arr(e, "img"), txt = arr(e, "txt")
            for i in 0..<nd {
                let d = try await run("Double\(i)", ["img": img, "txt": txt, "mod_img": arr(e, "mod_img"),
                                                      "mod_txt": arr(e, "mod_txt"), "rope_cos": cos, "rope_sin": sin])
                img = arr(d, "img_out"); txt = arr(d, "txt_out")
            }
            var hidden = MLMultiArray(concatenating: [txt, img], axis: 1, dataType: .float32)
            for i in 0..<ns {
                hidden = arr(try await run("Single\(i)", ["hidden": hidden, "mod": arr(e, "mod_single"),
                                                          "rope_cos": cos, "rope_sin": sin]), "hidden_out")
            }
            let noise = arr(try await run("Final", ["hidden": hidden, "temb": arr(e, "temb")]), "noise")
            let dsig = Float(sigmas[s + 1] - sigmas[s])
            var bad = 0
            for k in 0..<latent.count {
                let v = noise[k].floatValue
                if !v.isFinite { bad += 1 }
                lp[k] += dsig * v
            }
            firstLoad = false
            let dt = Date().timeIntervalSince(ts)
            stepTimes.append(dt)
            func avg(_ k: String) -> Double { let a = perKind[k] ?? [0]; return a.reduce(0, +) / Double(a.count) }
            emit(String(format: "step %d/%d total=%.2fs load=%.2fs predict=%.2fs double=%.0fms single=%.0fms nonfinite=%d",
                        s + 1, sigmas.count - 1, dt, loadSum, predSum, avg("Double"), avg("Single"), bad))
            emit(memLine("after step \(s + 1)"))
        }
        emit(String(format: "Klein %d steps total=%.1fs (warm steps avg %.2fs)", stepTimes.count, stepTimes.reduce(0, +),
                    stepTimes.dropFirst().reduce(0, +) / Double(max(stepTimes.count - 1, 1))))
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try Data(bytes: latent.dataPointer, count: latent.count * 4).write(to: docs.appending(path: "klein_latent.bin"))
        emit("saved Documents/klein_latent.bin")
        emit("RESULT OK \(chain)")
    } catch {
        emit("ERROR \(chain): \(error)")
        emit(memLine("at error"))
    }
    return out.joined(separator: "\n")
}

/// Where does the chain overhead come from? One Klein single-stream stage, three ways:
///  A  load once, same float32 inputs every time (the isolated-probe number)
///  B  reload the model before every prediction (what the per-stage pipeline does)
///  C  load once, feed each output back as the next input (Core ML-owned output buffers as inputs)
func runKleinBench(_ chain: String, stage: String, n: Int) async -> String {
    var out: [String] = []
    func emit(_ s: String) { log(s); out.append(s) }
    guard let dir = Bundle.main.url(forResource: chain, withExtension: nil, subdirectory: "Models") else {
        emit("chain \(chain) not found"); return ""
    }
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    let url = dir.appending(path: "\(stage).mlmodelc")
    do {
        func fill(_ shape: [NSNumber], _ scale: Float) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: shape, dataType: .float32)
            let p = a.dataPointer.assumingMemoryBound(to: Float.self)
            for k in 0..<a.count { p[k] = Float.random(in: -1...1) * scale }
            return a
        }
        var model = try MLModel(contentsOf: url, configuration: config)
        var feed: [String: Any] = [:]
        for (name, d) in model.modelDescription.inputDescriptionsByName {
            feed[name] = try fill(d.multiArrayConstraint!.shape, name == "mod" ? 0.1 : 1)
        }
        func ms(_ body: () async throws -> Void) async rethrows -> Double {
            let t = Date(); try await body(); return Date().timeIntervalSince(t) * 1000
        }
        func stats(_ a: [Double]) -> String {
            String(format: "avg=%.0fms min=%.0fms max=%.0fms n=%d", a.reduce(0, +) / Double(a.count), a.min()!, a.max()!, a.count)
        }
        var r = try await model.prediction(from: MLDictionaryFeatureProvider(dictionary: feed))   // warm-up
        let o0 = r.featureValue(for: "hidden_out")!.multiArrayValue!
        emit("bench stage=\(stage) outputType=\(o0.dataType.rawValue) (65568=float32, 65552=float16) hasPixelBuffer=\(o0.pixelBuffer != nil)")
        var a: [Double] = [], b: [Double] = [], bl: [Double] = [], c: [Double] = []
        for _ in 0..<n {
            a.append(try await ms { r = try await model.prediction(from: MLDictionaryFeatureProvider(dictionary: feed)) })
        }
        emit("A same-model-same-input  " + stats(a))
        for _ in 0..<n {
            bl.append(try await ms { model = try MLModel(contentsOf: url, configuration: config) })
            b.append(try await ms { r = try await model.prediction(from: MLDictionaryFeatureProvider(dictionary: feed)) })
        }
        emit("B reload-each-time       " + stats(b) + "   reload " + stats(bl))
        var f2 = feed
        for _ in 0..<n {
            f2["hidden"] = r.featureValue(for: "hidden_out")!.multiArrayValue!
            c.append(try await ms { r = try await model.prediction(from: MLDictionaryFeatureProvider(dictionary: f2)) })
        }
        emit("C output-fed-back        " + stats(c))
        // D: several different stages kept resident, run round-robin (no loads between predictions)
        // E: the same stages loaded fresh before each prediction and released after (the pipeline's pattern)
        model = try MLModel(contentsOf: url, configuration: config)   // keep only one instance alive
        let names = (0..<6).map { "Single\($0)" }
        var d: [Double] = [], e: [Double] = [], el: [Double] = []
        // (Keeping six 1024 stages resident at once gets the app killed by jetsam, so D keeps just two.)
        do {
            let resident = try names.prefix(2).map { try MLModel(contentsOf: dir.appending(path: "\($0).mlmodelc"), configuration: config) }
            for m in resident { _ = try await m.prediction(from: MLDictionaryFeatureProvider(dictionary: feed)) }
            for _ in 0..<4 {
                for m in resident {
                    d.append(try await ms { r = try await m.prediction(from: MLDictionaryFeatureProvider(dictionary: feed)) })
                }
            }
        }
        emit("D two-resident-alternating  " + stats(d))
        for _ in 0..<2 {
            for nm in names {
                var m: MLModel? = nil
                el.append(try await ms { m = try MLModel(contentsOf: dir.appending(path: "\(nm).mlmodelc"), configuration: config) })
                e.append(try await ms { r = try await m!.prediction(from: MLDictionaryFeatureProvider(dictionary: feed)) })
                m = nil
            }
        }
        emit("E six-load-run-release     " + stats(e) + "   load " + stats(el))
        emit(memLine("after D/E"))
        emit("thermal=\(ProcessInfo.processInfo.thermalState.rawValue) (0 nominal, 1 fair, 2 serious, 3 critical)")
        emit("RESULT OK kbench")
    } catch { emit("ERROR kbench: \(error)") }
    return out.joined(separator: "\n")
}

@main
struct ANEProbeApp: App {
    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "--kbench"), i + 2 < args.count {
            let chain = args[i + 1], stage = args[i + 2]
            Task.detached {
                _ = await runKleinBench(chain, stage: stage, n: 8)
                log("DONE")
                exit(0)
            }
        } else if let i = args.firstIndex(of: "--klein"), i + 1 < args.count {
            let name = args[i + 1]
            Task.detached {
                _ = await runKlein(name)
                log("DONE")
                exit(0)
            }
        } else if let i = args.firstIndex(of: "--chain"), i + 1 < args.count {
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
