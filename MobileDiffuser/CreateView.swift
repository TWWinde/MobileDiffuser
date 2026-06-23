// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI

/// Generation workspace: a full-bleed result canvas above a prompt + controls panel.
struct CreateView: View {
    @Bindable var model: AppModel

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            VStack(spacing: 0) {
                modelBar
                canvas
                controls
            }
        }
    }

    private var modelBar: some View {
        Button { model.tab = .models } label: {
            HStack(spacing: 10) {
                Image(systemName: "cube.box.fill").foregroundStyle(Theme.accent)
                Text(model.selected.displayName).font(.subheadline.weight(.semibold)).foregroundStyle(.primary)
                FitBadge(capabilities: model.capabilities(for: model.selected))
                Spacer()
                Text(model.statusText).font(.caption2)
                    .foregroundStyle(model.isFailed ? .red : .secondary).lineLimit(1)
                Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 16).padding(.vertical, 10)
            .background(Theme.surface)
        }
        .buttonStyle(.plain)
    }

    private var canvas: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 20, style: .continuous).fill(Theme.surface)
            if let cg = model.image {
                Image(decorative: cg, scale: 1).resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
            } else {
                VStack(spacing: 14) {
                    Image(systemName: model.isBusy ? "sparkles" : "photo.artframe")
                        .font(.system(size: 40, weight: .light)).foregroundStyle(.secondary)
                        .symbolEffect(.pulse, isActive: model.isBusy)
                    Text(model.isBusy ? model.statusText : "Describe an image, then Generate")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }
            if case .generating(let step, let total) = model.phase {
                VStack { Spacer()
                    ProgressView(value: Double(step), total: Double(total)).tint(Theme.accent).padding() }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).padding(16)
    }

    private var controls: some View {
        VStack(spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                TextField("Prompt", text: $model.prompt, axis: .vertical)
                    .textFieldStyle(.plain).lineLimit(1...3)
                    .padding(12)
                    .background(Theme.surface2, in: RoundedRectangle(cornerRadius: 12))
                Button { Task { await model.generate() } } label: {
                    Image(systemName: "arrow.up").font(.headline).frame(width: 44, height: 44)
                }
                .buttonStyle(.borderedProminent).tint(Theme.accent).clipShape(Circle())
                .disabled(model.isBusy || model.prompt.isEmpty)
            }
            HStack(spacing: 10) {
                menu("Size", $model.size, [512, 768, 1024])
                menu("Steps", $model.steps, [4, 8, 16])
                TextField("Seed", text: $model.seedText).frame(width: 64).textFieldStyle(.roundedBorder)
                Spacer()
            }
        }
        .padding(16).background(Theme.surface)
    }

    private func menu(_ label: String, _ sel: Binding<Int>, _ options: [Int]) -> some View {
        Picker(label, selection: sel) { ForEach(options, id: \.self) { Text("\($0)").tag($0) } }
            .pickerStyle(.menu).tint(.secondary)
    }
}
