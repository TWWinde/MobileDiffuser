// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI

/// Z-Image creation studio (dark theme). A result canvas above a prompt + controls panel; runs
/// `ZImagePipeline` via `AppModel`. The first non-FLUX model on the rebuilt MLX stack.
struct ContentView: View {
    @State private var model = AppModel()

    var body: some View {
        ZStack {
            Color(white: 0.06).ignoresSafeArea()
            VStack(spacing: 0) {
                canvas
                controls
            }
        }
        .tint(.orange)
        .preferredColorScheme(.dark)
    }

    private var canvas: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color(white: 0.10))
            if let cg = model.image {
                Image(decorative: cg, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
            } else {
                VStack(spacing: 14) {
                    Image(systemName: model.isBusy ? "sparkles" : "photo.artframe")
                        .font(.system(size: 40, weight: .light))
                        .foregroundStyle(.secondary)
                        .symbolEffect(.pulse, isActive: model.isBusy)
                    Text(model.isBusy ? model.statusText : "Describe an image, then Generate")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            if case .generating(let step, let total) = model.phase {
                VStack {
                    Spacer()
                    ProgressView(value: Double(step), total: Double(total))
                        .tint(.orange)
                        .padding()
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(16)
    }

    private var controls: some View {
        VStack(spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                TextField("Prompt", text: $model.prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...3)
                    .padding(12)
                    .background(Color(white: 0.12), in: RoundedRectangle(cornerRadius: 12))
                Button(action: { Task { await model.generate() } }) {
                    Image(systemName: "arrow.up")
                        .font(.headline)
                        .frame(width: 44, height: 44)
                }
                .buttonStyle(.borderedProminent)
                .clipShape(Circle())
                .disabled(model.isBusy || model.prompt.isEmpty)
            }
            HStack(spacing: 10) {
                picker("Size", selection: $model.size, options: [512, 768, 1024]) { "\($0)" }
                picker("Steps", selection: $model.steps, options: [4, 8, 16]) { "\($0)" }
                TextField("Seed", text: $model.seedText)
                    .frame(width: 70)
                    .textFieldStyle(.roundedBorder)
                Spacer()
                Text(model.statusText)
                    .font(.caption)
                    .foregroundStyle(model.isFailed ? .red : .secondary)
                    .lineLimit(1)
            }
            TextField("Model directory", text: $model.modelDirectory)
                .textFieldStyle(.roundedBorder)
                .font(.caption)
        }
        .padding(16)
        .background(Color(white: 0.09))
    }

    private func picker(_ label: String, selection: Binding<Int>, options: [Int],
                        format: @escaping (Int) -> String) -> some View {
        Picker(label, selection: selection) {
            ForEach(options, id: \.self) { Text(format($0)).tag($0) }
        }
        .pickerStyle(.menu)
        .tint(.secondary)
    }
}

private extension AppModel {
    var isFailed: Bool { if case .failed = phase { return true } else { return false } }
}

#Preview {
    ContentView()
}
