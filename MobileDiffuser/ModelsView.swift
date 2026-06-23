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
            VStack(spacing: Theme.Space.md) {
                ForEach(model.models) { m in
                    ModelCard(model: model, m: m) { detail = m }
                }
            }
            .padding(Theme.Space.lg)
        }
        .background(Theme.bg)
        .sheet(item: $detail) { m in ModelDetail(model: model, item: m) }
    }
}

/// Model management presented as a sheet — from the Create toolbar / model bar and from Settings.
/// Wraps `ModelsView` with a title bar and a Done button.
struct ModelsSheet: View {
    @Bindable var model: AppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ModelsView(model: model)
                .navigationTitle("Models")
                #if os(iOS)
                .navigationBarTitleDisplayMode(.inline)
                #endif
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
        }
        .tint(Theme.accent)
        #if os(macOS)
        .frame(minWidth: 540, minHeight: 600)
        #endif
    }
}

/// A single model card: title, fit badge, summary, family/precision/size chips, component bar,
/// and the install/use action.
struct ModelCard: View {
    @Bindable var model: AppModel
    let m: DiffusionModel
    let onDetails: () -> Void

    private var selected: Bool { model.selectedID == m.id }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            HStack {
                Text(m.displayName).font(.headline).foregroundStyle(Theme.textPrimary)
                Spacer()
                FitBadge(capabilities: model.capabilities(for: m))
            }
            Text("\(m.publisher) · \(m.summary)")
                .font(.caption).foregroundStyle(Theme.textSecondary).lineLimit(2)
            HStack(spacing: Theme.Space.xs) {
                Chip(text: m.family == .flux2 ? "FLUX.2" : "Z-Image")
                Chip(text: m.variants[0].precision.label, filled: true)
                Chip(text: ByteCountFormatter.string(fromByteCount: m.variants[0].approximateBytes, countStyle: .file))
                Spacer()
            }
            ComponentBar(components: m.variants[0].components)
            HStack(spacing: Theme.Space.sm) {
                ModelAction(model: model, m: m)
                Spacer()
                Button("Details", action: onDetails)
                    .font(.caption.weight(.semibold)).buttonStyle(.plain).foregroundStyle(Theme.accent)
            }
        }
        .studioCard()
        .overlay(RoundedRectangle(cornerRadius: Theme.corner, style: .continuous)
            .strokeBorder(selected ? Theme.accent : .clear, lineWidth: 1.5))
        .contentShape(Rectangle())
        .onTapGesture { withAnimation(Motion.select) { model.selectedID = m.id } }
    }
}

/// The card's install/use control. Re-tapping after a failure retries (the control re-enables when
/// the phase leaves `.downloading`).
struct ModelAction: View {
    @Bindable var model: AppModel
    let m: DiffusionModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        let isThis = model.selectedID == m.id
        if isThis, case .downloading(let f) = model.phase {
            HStack(spacing: Theme.Space.xs) {
                ProgressView(value: f).frame(width: 90).tint(Theme.accent)
                Text("\(Int(f * 100))%").font(.caption2).monospacedDigit().foregroundStyle(Theme.textSecondary)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Downloading")
            .accessibilityValue("\(Int(f * 100)) percent")
        } else if model.isDownloaded(m) || (m.family != .zImage) {
            Button {
                model.selectedID = m.id; dismiss()   // pick this model and return to Create
            } label: {
                Label(model.isDownloaded(m) ? "Use" : "Use (downloads on first run)", systemImage: "wand.and.stars")
            }
            .buttonStyle(StudioButtonStyle(.primary))
        } else {
            Button {
                model.selectedID = m.id; Task { await model.download() }
            } label: {
                Label("Download", systemImage: "arrow.down.circle")
            }
            .buttonStyle(StudioButtonStyle(.secondary)).disabled(model.isBusy)
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
            VStack(alignment: .leading, spacing: Theme.Space.lg) {
                HStack {
                    VStack(alignment: .leading) {
                        Text(item.displayName).font(.title2.weight(.semibold)).foregroundStyle(Theme.textPrimary)
                        Text("\(item.publisher) · \(item.license.label)").font(.caption).foregroundStyle(Theme.textSecondary)
                    }
                    Spacer()
                    Button { dismiss() } label: {
                        Image(systemName: "xmark.circle.fill").foregroundStyle(Theme.textSecondary)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Close")
                }
                FitBadge(capabilities: model.capabilities(for: item))
                Text(model.capabilities(for: item).note).font(.caption).foregroundStyle(Theme.textSecondary)

                VStack(alignment: .leading, spacing: Theme.Space.sm) {
                    row("Precision", v.precision.label)
                    row("Download size", ByteCountFormatter.string(fromByteCount: v.approximateBytes, countStyle: .file))
                    row("Transformer", ByteCountFormatter.string(fromByteCount: v.components.transformer, countStyle: .file))
                    row("Text encoder", ByteCountFormatter.string(fromByteCount: v.components.textEncoder, countStyle: .file))
                    row("VAE", ByteCountFormatter.string(fromByteCount: v.components.vae, countStyle: .file))
                }.studioCard()

                if model.selectedID == item.id, case .downloading(let f) = model.phase {
                    // Mirror the card's progress state so the detail reflects an in-flight download.
                    VStack(spacing: Theme.Space.xs) {
                        ProgressView(value: f).tint(Theme.accent)
                        Text("Downloading… \(Int(f * 100))%")
                            .font(.caption).monospacedDigit().foregroundStyle(Theme.textSecondary)
                    }
                    .frame(maxWidth: .infinity)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("Downloading")
                    .accessibilityValue("\(Int(f * 100)) percent")
                } else if item.family == .zImage && !model.isDownloaded(item) {
                    Button { model.selectedID = item.id; Task { await model.download() } } label: {
                        Label("Download", systemImage: "arrow.down.circle").frame(maxWidth: .infinity)
                    }.buttonStyle(StudioButtonStyle(.primary)).disabled(model.isBusy)
                } else {
                    Button { model.selectedID = item.id; dismiss() } label: {
                        Label("Use in Create", systemImage: "wand.and.stars").frame(maxWidth: .infinity)
                    }.buttonStyle(StudioButtonStyle(.primary))
                }
            }
            .padding(Theme.Space.xl)
        }
        .background(Theme.bg)
    }

    private func row(_ k: String, _ value: String) -> some View {
        HStack {
            Text(k).foregroundStyle(Theme.textSecondary)
            Spacer()
            Text(value).foregroundStyle(Theme.textPrimary).monospacedDigit()
        }.font(.subheadline)
    }
}
