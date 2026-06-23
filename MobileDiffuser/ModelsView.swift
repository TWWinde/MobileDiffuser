// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI
import DiffusionCore

/// Download center: family cards with precision chips, hardware-fit badges, and install/use.
struct ModelsView: View {
    @Bindable var model: AppModel
    @State private var detail: DiffusionModel?

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                ForEach(model.models) { card($0) }
            }
            .padding(16)
        }
        .background(Theme.bg)
        .sheet(item: $detail) { m in ModelDetail(model: model, item: m) }
    }

    private func card(_ m: DiffusionModel) -> some View {
        let selected = model.selectedID == m.id
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(m.displayName).font(.headline)
                Spacer()
                FitBadge(capabilities: model.capabilities(for: m))
            }
            Text("\(m.publisher) · \(m.summary)").font(.caption).foregroundStyle(.secondary).lineLimit(2)
            HStack(spacing: 6) {
                Chip(text: m.family == .flux2 ? "FLUX.2" : "Z-Image")
                Chip(text: m.variants[0].precision.label, filled: true)
                Chip(text: ByteCountFormatter.string(fromByteCount: m.variants[0].approximateBytes, countStyle: .file))
                Spacer()
            }
            HStack(spacing: 10) {
                action(m)
                Spacer()
                Button("Details") { detail = m }.font(.caption).buttonStyle(.plain).foregroundStyle(Theme.accent)
            }
        }
        .studioCard()
        .overlay(RoundedRectangle(cornerRadius: Theme.corner, style: .continuous)
            .strokeBorder(selected ? Theme.accent : .clear, lineWidth: 1.5))
        .contentShape(Rectangle())
        .onTapGesture { model.selectedID = m.id }
    }

    @ViewBuilder private func action(_ m: DiffusionModel) -> some View {
        let isThis = model.selectedID == m.id
        if isThis, case .downloading(let f) = model.phase {
            HStack(spacing: 6) { ProgressView(value: f).frame(width: 90)
                Text("\(Int(f * 100))%").font(.caption2).monospacedDigit().foregroundStyle(.secondary) }
        } else if model.isDownloaded(m) || (m.family != .zImage) {
            Button {
                model.selectedID = m.id; model.tab = .create
            } label: { Label(m.family == .zImage ? "Use" : "Use (downloads on first run)", systemImage: "wand.and.stars") }
                .buttonStyle(.borderedProminent).tint(Theme.accent).controlSize(.small)
        } else {
            Button {
                model.selectedID = m.id; Task { await model.download() }
            } label: { Label("Download", systemImage: "arrow.down.circle") }
                .buttonStyle(.bordered).controlSize(.small).disabled(model.isBusy)
        }
    }
}

/// Model detail: variant table, component breakdown, fit, install.
private struct ModelDetail: View {
    @Bindable var model: AppModel
    let item: DiffusionModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        let v = item.variants[0]
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    VStack(alignment: .leading) {
                        Text(item.displayName).font(.title2.weight(.semibold))
                        Text("\(item.publisher) · \(item.license.label)").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button { dismiss() } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
                        .buttonStyle(.plain)
                }
                FitBadge(capabilities: model.capabilities(for: item))
                Text(model.capabilities(for: item).note).font(.caption).foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 8) {
                    row("Precision", v.precision.label)
                    row("Download size", ByteCountFormatter.string(fromByteCount: v.approximateBytes, countStyle: .file))
                    row("Transformer", ByteCountFormatter.string(fromByteCount: v.components.transformer, countStyle: .file))
                    row("Text encoder", ByteCountFormatter.string(fromByteCount: v.components.textEncoder, countStyle: .file))
                    row("VAE", ByteCountFormatter.string(fromByteCount: v.components.vae, countStyle: .file))
                }.studioCard()

                if item.family == .zImage && !model.isDownloaded(item) {
                    Button { model.selectedID = item.id; Task { await model.download() } } label: {
                        Label("Download", systemImage: "arrow.down.circle").frame(maxWidth: .infinity)
                    }.buttonStyle(.borderedProminent).tint(Theme.accent).disabled(model.isBusy)
                } else {
                    Button { model.selectedID = item.id; model.tab = .create; dismiss() } label: {
                        Label("Use in Create", systemImage: "wand.and.stars").frame(maxWidth: .infinity)
                    }.buttonStyle(.borderedProminent).tint(Theme.accent)
                }
            }
            .padding(20)
        }
        .background(Theme.bg)
        .preferredColorScheme(.dark)
    }

    private func row(_ k: String, _ value: String) -> some View {
        HStack { Text(k).foregroundStyle(.secondary); Spacer(); Text(value).monospacedDigit() }.font(.subheadline)
    }
}
