// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI
import DiffusionCore
import AppEngines   // re-exports Flux2DiffusionEngine on macOS (precision enums); empty on iOS

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
        .sheet(item: $detail) { m in
            ModelDetail(model: model, item: m)
                #if os(macOS)
                .frame(minWidth: 520, idealWidth: 560, minHeight: 560, idealHeight: 680)
                #endif
        }
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
            #if os(macOS)
            let _ = model.componentsRevision   // re-render when the on-disk component set changes
            #endif
            HStack {
                Text(m.displayName).font(.headline).foregroundStyle(Theme.textPrimary)
                Spacer()
                FitBadge(capabilities: model.capabilities(for: m))
            }
            Text("\(m.publisher) · \(m.summary)")
                .font(.caption).foregroundStyle(Theme.textSecondary).lineLimit(2)
            HStack(spacing: Theme.Space.xs) {
                Chip(text: m.family == .flux2 ? "FLUX.2" : "Z-Image")
                #if os(macOS)
                if m.family == .flux2 {
                    Chip(text: model.fluxTransformer.label, filled: true)
                    Chip(text: ByteCountFormatter.string(fromByteCount: model.fluxActiveBytes, countStyle: .file))
                } else {
                    Chip(text: m.variants[0].precision.label, filled: true)
                    Chip(text: ByteCountFormatter.string(fromByteCount: m.variants[0].approximateBytes, countStyle: .file))
                }
                #else
                Chip(text: m.variants[0].precision.label, filled: true)
                Chip(text: ByteCountFormatter.string(fromByteCount: m.variants[0].approximateBytes, countStyle: .file))
                #endif
                Spacer()
            }
            #if os(macOS)
            if m.family == .flux2 {
                // The active recipe + a tri-state install status, so the Use/Download action is clear.
                let missing = model.fluxMissingCount
                let total = model.fluxActiveCount
                let status = missing == 0 ? "installed"
                    : (missing == total ? "not installed" : "\(missing) of \(total) missing")
                let color: Color = missing == 0 ? Theme.fitGreen
                    : (missing == total ? Theme.textTertiary : Theme.fitAmber)
                Text("Active: \(model.fluxRecipeLabel) · \(status)")
                    .font(.caption2).foregroundStyle(color)
            }
            #endif
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
        } else if model.isDownloaded(m) {
            Button {
                model.selectedID = m.id; dismiss()   // pick this model and return to Create
            } label: {
                Label("Use", systemImage: "wand.and.stars")
            }
            .buttonStyle(StudioButtonStyle(.primary))
        } else {
            Button {
                model.selectedID = m.id; Task { await model.download() }
            } label: {
                #if os(macOS)
                Label(m.family == .flux2 ? model.fluxDownloadActionLabel : "Download", systemImage: "arrow.down.circle")
                #else
                Label("Download", systemImage: "arrow.down.circle")
                #endif
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
    @State private var confirmDelete = false

    var body: some View {
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

                #if os(macOS)
                if item.family == .flux2 {
                    // FLUX: choose precision, then manage each weight component individually.
                    precisionSection
                    if model.selectedID == item.id, case .downloading(let f) = model.phase {
                        // A card-initiated download (completing the active recipe) is visible here too.
                        VStack(spacing: Theme.Space.xs) {
                            ProgressView(value: f).tint(Theme.accent)
                            Text("Downloading… \(Int(f * 100))%")
                                .font(.caption).monospacedDigit().foregroundStyle(Theme.textSecondary)
                        }.frame(maxWidth: .infinity)
                    }
                    componentsSection
                    if model.isDownloaded(item) {
                        Button { model.selectedID = item.id; dismiss() } label: {
                            Label("Use in Create", systemImage: "wand.and.stars").frame(maxWidth: .infinity)
                        }.buttonStyle(StudioButtonStyle(.primary))
                    }
                } else {
                    variantTable
                    installAction
                }
                #else
                variantTable
                installAction
                #endif
            }
            .padding(Theme.Space.xl)
        }
        .background(Theme.bg)
        .scrollBounceBehavior(.basedOnSize)
        .confirmationDialog("Delete \(item.displayName) weights?",
                            isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete", role: .destructive) { Task { await model.delete(item) } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Frees the disk space. You can download it again anytime.")
        }
    }

    private func row(_ k: String, _ value: String) -> some View {
        HStack {
            Text(k).foregroundStyle(Theme.textSecondary)
            Spacer()
            Text(value).foregroundStyle(Theme.textPrimary).monospacedDigit()
        }.font(.subheadline)
    }

    /// The fixed variant table (used for single-precision models like Z-Image).
    private var variantTable: some View {
        let v = item.variants[0]
        return VStack(alignment: .leading, spacing: Theme.Space.sm) {
            row("Precision", v.precision.label)
            row("Download size", ByteCountFormatter.string(fromByteCount: v.approximateBytes, countStyle: .file))
            row("Transformer", ByteCountFormatter.string(fromByteCount: v.components.transformer, countStyle: .file))
            row("Text encoder", ByteCountFormatter.string(fromByteCount: v.components.textEncoder, countStyle: .file))
            row("VAE", ByteCountFormatter.string(fromByteCount: v.components.vae, countStyle: .file))
        }.studioCard()
    }

    /// Model-level install action (download → progress → use + delete) for single-component models.
    @ViewBuilder private var installAction: some View {
        if model.selectedID == item.id, case .downloading(let f) = model.phase {
            VStack(spacing: Theme.Space.xs) {
                ProgressView(value: f).tint(Theme.accent)
                Text("Downloading… \(Int(f * 100))%")
                    .font(.caption).monospacedDigit().foregroundStyle(Theme.textSecondary)
            }
            .frame(maxWidth: .infinity)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Downloading")
            .accessibilityValue("\(Int(f * 100)) percent")
        } else if !model.isDownloaded(item) {
            Button { model.selectedID = item.id; Task { await model.download() } } label: {
                Label("Download", systemImage: "arrow.down.circle").frame(maxWidth: .infinity)
            }.buttonStyle(StudioButtonStyle(.primary)).disabled(model.isBusy)
        } else {
            VStack(spacing: Theme.Space.sm) {
                Button { model.selectedID = item.id; dismiss() } label: {
                    Label("Use in Create", systemImage: "wand.and.stars").frame(maxWidth: .infinity)
                }.buttonStyle(StudioButtonStyle(.primary))
                Button { confirmDelete = true } label: {
                    Label("Delete weights", systemImage: "trash").frame(maxWidth: .infinity)
                }.buttonStyle(StudioButtonStyle(.secondary)).disabled(model.isBusy)
            }
        }
    }

    #if os(macOS)
    /// FLUX precision: independent transformer + text-encoder precision pickers. Changing either
    /// re-points which weights are needed, so the install state below updates accordingly.
    @ViewBuilder private var precisionSection: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            Text("PRECISION").font(.caption2.weight(.semibold)).foregroundStyle(Theme.textTertiary)
            VStack(spacing: Theme.Space.sm) {
                precisionRow(title: "Model precision",
                             value: model.fluxTransformer.label, note: model.fluxTransformer.note) {
                    ForEach(Flux2FacadeEngine.FluxTransformerPrecision.allCases) { option in
                        Button(option.label) { withAnimation(Motion.select) { model.fluxTransformer = option } }
                    }
                }
                Divider().background(Theme.hairline)
                precisionRow(title: "Text encoder",
                             value: model.fluxEncoder.label, note: model.fluxEncoder.note) {
                    ForEach(Flux2FacadeEngine.FluxEncoderPrecision.allCases) { option in
                        Button(option.label) { withAnimation(Motion.select) { model.fluxEncoder = option } }
                    }
                }
            }
            .studioCard()
        }
    }

    @ViewBuilder private func precisionRow<Menu: View>(title: String, value: String, note: String,
                                                       @ViewBuilder menu: () -> Menu) -> some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.subheadline).foregroundStyle(Theme.textPrimary)
                Text(note).font(.caption2).foregroundStyle(Theme.textTertiary)
            }
            Spacer(minLength: Theme.Space.md)
            SwiftUI.Menu { menu() } label: {
                HStack(spacing: 4) {
                    Text(value).font(.subheadline).foregroundStyle(Theme.accent)
                    Image(systemName: "chevron.up.chevron.down").font(.caption2).foregroundStyle(Theme.textTertiary)
                }
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
        }
    }

    /// The per-component download/delete list — the user decides exactly which weights to keep.
    @ViewBuilder private var componentsSection: some View {
        let _ = model.componentsRevision   // re-read the on-disk list when it changes
        let components = model.fluxComponents()
        let onDisk = components.filter(\.isDownloaded).reduce(Int64(0)) { $0 + $1.bytes }
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            HStack {
                Text("COMPONENTS").font(.caption2.weight(.semibold)).foregroundStyle(Theme.textTertiary)
                Spacer()
                Text("\(ByteCountFormatter.string(fromByteCount: onDisk, countStyle: .file)) on disk")
                    .font(.caption2).foregroundStyle(Theme.textTertiary)
            }
            VStack(spacing: 0) {
                ForEach(Array(components.enumerated()), id: \.element.id) { index, component in
                    if index > 0 { Divider().background(Theme.hairline) }
                    componentRow(component)
                }
            }.studioCard()
        }
    }

    @ViewBuilder private func componentRow(_ c: Flux2FacadeEngine.Flux2ComponentInfo) -> some View {
        HStack(spacing: Theme.Space.md) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(c.title).font(.subheadline.weight(.medium)).foregroundStyle(Theme.textPrimary)
                    Text(c.kind.rawValue).font(.caption2)
                        .padding(.horizontal, 7).padding(.vertical, 2)
                        .background(badgeColor(c.kind).opacity(0.18), in: Capsule())
                        .foregroundStyle(badgeColor(c.kind))
                    if model.fluxActiveComponentIDs.contains(c.id) {
                        Text("Active").font(.caption2.weight(.medium))
                            .padding(.horizontal, 7).padding(.vertical, 2)
                            .background(Theme.fitGreen.opacity(0.18), in: Capsule())
                            .foregroundStyle(Theme.fitGreen)
                    }
                }
                if !c.subtitle.isEmpty {
                    Text(c.subtitle).font(.caption2).foregroundStyle(Theme.textTertiary)
                }
                Text(c.repo).font(.caption2.monospaced()).foregroundStyle(Theme.textTertiary)
                    .lineLimit(1).truncationMode(.middle)
            }
            Spacer(minLength: Theme.Space.sm)
            Text(ByteCountFormatter.string(fromByteCount: c.bytes, countStyle: .file))
                .font(.caption).foregroundStyle(Theme.textSecondary)
            componentControl(c).frame(width: 84, alignment: .trailing)
        }
        .frame(minHeight: 52)
        .padding(.vertical, 8)
    }

    @ViewBuilder private func componentControl(_ c: Flux2FacadeEngine.Flux2ComponentInfo) -> some View {
        if model.fluxComponentDownloadID == c.id {
            ProgressView(value: model.fluxComponentFraction).frame(width: 54).tint(Theme.accent)
        } else if model.fluxComponentError?.id == c.id {
            Button { Task { await model.downloadFluxComponent(c.id) } } label: {
                HStack(spacing: 4) { Image(systemName: "arrow.clockwise"); Text("Retry") }
                    .font(.caption.weight(.medium)).foregroundStyle(Theme.danger)
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .overlay(Capsule().strokeBorder(Theme.danger.opacity(0.5)))
            }
            .buttonStyle(.plain).disabled(model.isBusy)
            .accessibilityLabel("Retry \(c.title)")
        } else if c.isDownloaded {
            Button { Task { await model.deleteFluxComponent(c.id) } } label: {
                Image(systemName: "trash").foregroundStyle(Theme.textSecondary)
            }
            .buttonStyle(.plain).disabled(model.isBusy)
            .accessibilityLabel("Delete \(c.title)")
        } else {
            Button { Task { await model.downloadFluxComponent(c.id) } } label: {
                HStack(spacing: 4) { Image(systemName: "arrow.down"); Text("Get") }
                    .font(.caption.weight(.medium)).foregroundStyle(Theme.accent)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .overlay(Capsule().strokeBorder(Theme.accent.opacity(0.5)))
            }
            .buttonStyle(.plain).disabled(model.isBusy)
            .accessibilityLabel("Download \(c.title)")
        }
    }

    private func badgeColor(_ kind: Flux2FacadeEngine.Flux2ComponentInfo.Kind) -> Color {
        switch kind {
        case .transformer: return Theme.accent
        case .textEncoder: return Color(red: 0.35, green: 0.6, blue: 0.9)
        case .vae: return Theme.textTertiary
        }
    }
    #endif
}
