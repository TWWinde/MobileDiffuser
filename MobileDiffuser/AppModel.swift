// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI
import CoreGraphics
import ZImageMLX

/// Drives a single Z-Image generation through `ZImagePipeline`. UI state lives on the main actor;
/// the heavy denoise runs on a detached task so the UI stays responsive.
@MainActor
@Observable
final class AppModel {
    enum Phase: Equatable {
        case idle
        case loading(Double)        // 0…1 model load
        case generating(Int, Int)   // step, total
        case done
        case failed(String)
    }

    /// Folder laid out like `Z-Image-Turbo-6B-MLX-Q4` (text_encoder/ transformer/ vae/ tokenizer/).
    var modelDirectory = "\(NSHomeDirectory())/code/z-image-weights"
    var prompt = "a red panda on a mossy rock, soft morning light"
    var size = 1024
    var steps = 8
    var seedText = "42"
    var phase: Phase = .idle
    var image: CGImage?

    private var pipeline: ZImagePipeline?
    private var loadedDirectory: String?

    var isBusy: Bool {
        switch phase { case .loading, .generating: return true; default: return false }
    }

    var statusText: String {
        switch phase {
        case .idle: return "Ready"
        case .loading(let f): return "Loading model… \(Int(f * 100))%"
        case .generating(let s, let t): return "Generating… step \(s)/\(t)"
        case .done: return "Done"
        case .failed(let m): return "Failed: \(m)"
        }
    }

    func generate() async {
        let directory = modelDirectory
        let prompt = self.prompt, size = self.size, steps = self.steps
        let seed = UInt64(seedText) ?? 42
        do {
            if pipeline == nil || loadedDirectory != directory {
                phase = .loading(0)
                let loaded = ZImagePipeline(modelDirectory: URL(fileURLWithPath: directory))
                try await loaded.loadModels { fraction in
                    Task { @MainActor in self.phase = .loading(fraction) }
                }
                pipeline = loaded
                loadedDirectory = directory
            }
            guard let pipeline else { return }
            phase = .generating(0, steps)
            let cgImage = try await Task.detached(priority: .userInitiated) {
                try pipeline.generate(prompt: prompt, size: size, steps: steps, seed: seed) { step, total in
                    Task { @MainActor in self.phase = .generating(step, total) }
                }
            }.value
            image = cgImage
            phase = .done
        } catch {
            phase = .failed(String(describing: error))
        }
    }
}
