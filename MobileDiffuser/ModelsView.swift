// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI
import DiffusionCore

/// Download center: model cards with precision chips, hardware-fit badges, and install/use.
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
                    ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
                }
        }
        .tint(Theme.accent)
        #if os(macOS)
        .frame(minWidth: 540, minHeight: 600)
        #endif
    }
}

/// A single model card: title, fit badge, summary, family/precision/size chips, install state.
/// Recipe-driven, so every family renders the same anatomy.
struct ModelCard: View {
    @Bindable var model: AppModel
    let m: DiffusionModel
    let onDetails: () -> Void

    private var selected: Bool { model.selectedID == m.id }

    var body: some View {
        let _ = model.componentsRevision   // re-render when on-disk install state changes
        let recipe = model.recipe(for: m)
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            HStack {
                Text(m.displayName).font(.headline).foregroundStyle(Theme.textPrimary)
                    .lineLimit(1).minimumScaleFactor(0.7).layoutPriority(1)
                Spacer()
                FitBadge(capabilities: model.capabilities(for: m))
            }
            Text("\(m.publisher) · \(m.summary)")
                .font(.caption).foregroundStyle(Theme.textSecondary).lineLimit(2)
            HStack(spacing: Theme.Space.xs) {
                Chip(text: m.family == .flux2 ? "FLUX.2" : "Z-Image")
                Chip(text: recipe.axes.first?.selectedOption?.label ?? m.variants[0].precision.label, filled: true)
                Chip(text: ByteCountFormatter.string(fromByteCount: recipe.activeBytes, countStyle: .file))
                Spacer()
            }
            installStatus(recipe)
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

    /// Tri-state install line: prefixed with the active recipe when the model has precision axes.
    @ViewBuilder private func installStatus(_ recipe: ModelRecipe) -> some View {
        let missing = recipe.missing.count
        let total = recipe.activeCount
        if total > 0 {
            let status = missing == 0 ? "installed"
                : (missing == total ? "not installed" : "\(missing) of \(total) missing")
            let color: Color = missing == 0 ? Theme.fitGreen
                : (missing == total ? Theme.textTertiary : Theme.fitAmber)
            let recipeText = recipe.axes.map { $0.selectedOption?.label ?? "" }.joined(separator: " · ")
            Text(recipe.axes.isEmpty ? status : "\(recipeText) · \(status)")
                .font(.caption2).foregroundStyle(color)
                .lineLimit(2).minimumScaleFactor(0.85)
        }
    }
}

/// The card's install/use control. Use when installed, otherwise a self-describing Download.
struct ModelAction: View {
    @Bindable var model: AppModel
    let m: DiffusionModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        let recipe = model.recipe(for: m)
        if model.selectedID == m.id, case .downloading(let f) = model.phase {
            VStack(alignment: .trailing, spacing: 2) {
                HStack(spacing: Theme.Space.xs) {
                    ProgressView(value: f).frame(width: 90).tint(Theme.accent)
                    Text("\(Int(f * 100))%").font(.caption2).monospacedDigit().foregroundStyle(Theme.textSecondary)
                }
                if let detail = model.downloadMeter.compactDetail {
                    Text(detail).font(.caption2.monospacedDigit()).foregroundStyle(Theme.textTertiary)
                        .lineLimit(1).minimumScaleFactor(0.6)
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Downloading").accessibilityValue(model.downloadMeter.detail ?? "\(Int(f * 100)) percent")
        } else if recipe.isInstalled {
            Button { model.selectedID = m.id; dismiss() } label: {
                Label("Use", systemImage: "wand.and.stars")
            }
            .buttonStyle(StudioButtonStyle(.primary))
        } else {
            Button { model.startInstallRecipe(m) } label: {
                Label(downloadLabel(recipe), systemImage: "arrow.down.circle")
            }
            .buttonStyle(StudioButtonStyle(.secondary)).disabled(model.isBusy)
        }
    }

    private func downloadLabel(_ recipe: ModelRecipe) -> String {
        let bytes = ByteCountFormatter.string(fromByteCount: recipe.missingBytes, countStyle: .file)
        return recipe.missing.count < recipe.activeCount ? "Complete · \(bytes)" : "Download · \(bytes)"
    }
}

/// One data-driven detail template for every model: header, fit, optional precision axes, a
/// per-component install/delete list, and a footer (Use / download-all / remove-all).
private struct ModelDetail: View {
    @Bindable var model: AppModel
    let item: DiffusionModel
    @Environment(\.dismiss) private var dismiss
    @State private var confirmRemoveAll = false
    @State private var pendingDelete: RecipeComponent?

    private var isDownloadingModel: Bool {
        if model.selectedID == item.id, case .downloading = model.phase { return true }
        return false
    }

    var body: some View {
        let _ = model.componentsRevision
        let recipe = model.recipe(for: item)
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.lg) {
                header
                FitBadge(capabilities: model.capabilities(for: item))
                Text(model.capabilities(for: item).note).font(.caption).foregroundStyle(Theme.textSecondary)

                if !recipe.axes.isEmpty { precisionSection(recipe.axes) }
                if isDownloadingModel, case .downloading(let f) = model.phase { modelProgress(f) }
                componentsSection(recipe)
                footer(recipe)
            }
            .padding(Theme.Space.xl)
        }
        .background(Theme.bg)
        .scrollBounceBehavior(.basedOnSize)
        // Centered alerts (not confirmationDialog) — on iOS the dialog can render as a popover anchored to
        // the wrong row; an alert is always a centered modal, which is also fine for a destructive confirm.
        .alert("Remove all \(item.displayName) weights?", isPresented: $confirmRemoveAll) {
            Button("Remove all", role: .destructive) { Task { await model.delete(item) } }
            Button("Cancel", role: .cancel) {}
        } message: { Text("Frees the disk space. You can download it again anytime.") }
        .alert(pendingDelete.map { "Delete \($0.title)?" } ?? "",
               isPresented: Binding(get: { pendingDelete != nil },
                                    set: { if !$0 { pendingDelete = nil } }),
               presenting: pendingDelete) { c in
            Button("Delete", role: .destructive) { Task { await model.removeComponent(c.id, model: item) } }
            Button("Cancel", role: .cancel) {}
        } message: { c in
            Text(c.isActive ? "This is part of the active recipe — you'll need it to generate."
                            : "Frees the disk space. You can download it again anytime.")
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(item.displayName).font(.title2.weight(.semibold)).foregroundStyle(Theme.textPrimary)
                Text("\(item.publisher) · \(item.license.label)").font(.caption).foregroundStyle(Theme.textSecondary)
            }
            Spacer()
            Button { dismiss() } label: {
                Image(systemName: "xmark.circle.fill").foregroundStyle(Theme.textSecondary)
            }
            .buttonStyle(.plain).accessibilityLabel("Close")
        }
    }

    @ViewBuilder private func modelProgress(_ f: Double) -> some View {
        VStack(spacing: Theme.Space.xs) {
            ProgressView(value: f).tint(Theme.accent)
            // Live bytes / speed / ETA when the meter has it, else the plain percentage.
            Text(model.downloadMeter.detail ?? "Downloading… \(Int(f * 100))%")
                .font(.caption).monospacedDigit().foregroundStyle(Theme.textSecondary)
                .lineLimit(1).minimumScaleFactor(0.7)
        }.frame(maxWidth: .infinity)
    }

    // MARK: Precision

    @ViewBuilder private func precisionSection(_ axes: [PrecisionAxis]) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            Text("PRECISION").font(.caption2.weight(.semibold)).foregroundStyle(Theme.textTertiary)
            VStack(spacing: Theme.Space.sm) {
                ForEach(Array(axes.enumerated()), id: \.element.id) { index, axis in
                    if index > 0 { Divider().background(Theme.hairline) }
                    precisionRow(axis)
                }
            }.studioCard()
        }
    }

    @ViewBuilder private func precisionRow(_ axis: PrecisionAxis) -> some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(axis.title).font(.subheadline).foregroundStyle(Theme.textPrimary)
                if let note = axis.selectedOption?.note, !note.isEmpty {
                    Text(note).font(.caption2).foregroundStyle(Theme.textTertiary).lineLimit(2)
                }
            }
            Spacer(minLength: Theme.Space.md)
            Menu {
                ForEach(axis.options) { option in
                    Button(option.label) { withAnimation(Motion.select) { model.setPrecision(axisID: axis.id, optionID: option.id) } }
                }
            } label: {
                HStack(spacing: 4) {
                    Text(axis.selectedOption?.label ?? "").font(.subheadline).foregroundStyle(Theme.accent)
                    Image(systemName: "chevron.up.chevron.down").font(.caption2).foregroundStyle(Theme.textTertiary)
                }
            }
            .fixedSize()
        }
    }

    // MARK: Components

    @ViewBuilder private func componentsSection(_ recipe: ModelRecipe) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            HStack {
                Text("COMPONENTS").font(.caption2.weight(.semibold)).foregroundStyle(Theme.textTertiary)
                Spacer()
                Text("\(ByteCountFormatter.string(fromByteCount: recipe.bytesOnDisk, countStyle: .file)) on disk")
                    .font(.caption2).foregroundStyle(Theme.textTertiary)
                    .lineLimit(1).fixedSize()
            }
            VStack(spacing: 0) {
                ForEach(Array(recipe.components.enumerated()), id: \.element.id) { index, c in
                    if index > 0 { Divider().background(Theme.hairline) }
                    componentRow(c, showActive: !recipe.axes.isEmpty)
                }
            }.studioCard()
        }
    }

    @ViewBuilder private func componentRow(_ c: RecipeComponent, showActive: Bool) -> some View {
        HStack(spacing: Theme.Space.md) {
            VStack(alignment: .leading, spacing: 3) {
                // Title on its own full-width line (no longer competing with the badges, which forced it
                // to wrap mid-word — "Klein 4B · 4-\nbit"); badges sit on the line below.
                Text(c.title).font(.subheadline.weight(.medium)).foregroundStyle(Theme.textPrimary)
                    .lineLimit(1).minimumScaleFactor(0.6)
                HStack(spacing: 6) {
                    Text(c.kind.rawValue).font(.caption2)
                        .padding(.horizontal, 7).padding(.vertical, 2)
                        .background(badgeColor(c.kind).opacity(0.18), in: Capsule())
                        .foregroundStyle(badgeColor(c.kind))
                    if showActive && c.isActive {
                        Text("Active").font(.caption2.weight(.medium))
                            .padding(.horizontal, 7).padding(.vertical, 2)
                            .background(Theme.accentSoft, in: Capsule())
                            .foregroundStyle(Theme.accent)
                    }
                }
                if !c.subtitle.isEmpty {
                    Text(c.subtitle).font(.caption2).foregroundStyle(Theme.textTertiary)
                }
                // Show the FULL HuggingFace id — shrink to fit one line rather than middle-truncating it
                // to "black-fores…klein-4B" (middle-truncation kept only as a last resort past 0.6 scale).
                Text(c.repo).font(.caption2.monospaced()).foregroundStyle(Theme.textTertiary)
                    .lineLimit(1).minimumScaleFactor(0.6).truncationMode(.middle)
                // While THIS component is downloading, show its live bytes / speed / ETA (the shared meter
                // tracks the one active download). componentProgress is non-nil only for the active id.
                if model.componentProgress(c.id, model: item) != nil, let detail = model.downloadMeter.compactDetail {
                    Text(detail).font(.caption2.monospacedDigit().weight(.medium)).foregroundStyle(Theme.accent)
                        .lineLimit(1).minimumScaleFactor(0.6)   // shrink to fit, never truncate
                }
            }
            Spacer(minLength: Theme.Space.sm)
            Text(ByteCountFormatter.string(fromByteCount: c.bytes, countStyle: .file))
                .font(.caption).foregroundStyle(Theme.textSecondary)
                .lineLimit(1).fixedSize()   // "2.18 GB" never wraps — keep it from squeezing the title
            componentControl(c).frame(width: 84, alignment: .trailing)
        }
        .frame(minHeight: 52)
        .padding(.vertical, 8)
    }

    @ViewBuilder private func componentControl(_ c: RecipeComponent) -> some View {
        if let f = model.componentProgress(c.id, model: item) {
            ProgressView(value: f).frame(width: 54).tint(Theme.accent)
        } else if model.componentErrorMessage(c.id) != nil {
            Button { model.startInstallComponent(c.id, model: item) } label: {
                pill("Retry", icon: "arrow.clockwise", color: Theme.danger)
            }
            .buttonStyle(.plain).disabled(model.isBusy).accessibilityLabel("Retry \(c.title)")
        } else if c.isDownloaded {
            Button { pendingDelete = c } label: {
                Image(systemName: "trash").foregroundStyle(Theme.textSecondary)
            }
            .buttonStyle(.plain).disabled(model.isBusy).accessibilityLabel("Delete \(c.title)")
        } else {
            Button { model.startInstallComponent(c.id, model: item) } label: {
                pill("Get", icon: "arrow.down", color: Theme.accent)
            }
            .buttonStyle(.plain).disabled(model.isBusy).accessibilityLabel("Download \(c.title)")
        }
    }

    private func pill(_ text: String, icon: String, color: Color) -> some View {
        HStack(spacing: 4) { Image(systemName: icon); Text(text) }
            .font(.caption.weight(.medium)).foregroundStyle(color)
            .padding(.horizontal, 10).padding(.vertical, 6)
            .overlay(Capsule().strokeBorder(color.opacity(0.5)))
    }

    // MARK: Footer

    @ViewBuilder private func footer(_ recipe: ModelRecipe) -> some View {
        VStack(spacing: Theme.Space.sm) {
            if recipe.isInstalled {
                Button { model.selectedID = item.id; dismiss() } label: {
                    Label("Use in Create", systemImage: "wand.and.stars").frame(maxWidth: .infinity)
                }.buttonStyle(StudioButtonStyle(.primary))
            } else if recipe.components.count > 1 && !isDownloadingModel {
                Button { model.startInstallRecipe(item) } label: {
                    Label(footerDownloadLabel(recipe), systemImage: "arrow.down.circle").frame(maxWidth: .infinity)
                }.buttonStyle(StudioButtonStyle(.primary)).disabled(model.isBusy)
            }
            if recipe.components.count > 1 && recipe.bytesOnDisk > 0 {
                Button { confirmRemoveAll = true } label: {
                    Label("Remove all weights", systemImage: "trash").frame(maxWidth: .infinity)
                }.buttonStyle(StudioButtonStyle(.secondary)).disabled(model.isBusy)
            }
        }
    }

    private func footerDownloadLabel(_ recipe: ModelRecipe) -> String {
        let bytes = ByteCountFormatter.string(fromByteCount: recipe.missingBytes, countStyle: .file)
        return recipe.missing.count < recipe.activeCount ? "Complete recipe · \(bytes)" : "Download all · \(bytes)"
    }

    private func badgeColor(_ kind: RecipeComponent.Kind) -> Color {
        switch kind {
        case .transformer, .weights: return Theme.accent
        case .textEncoder: return Color(red: 0.35, green: 0.6, blue: 0.9)
        case .vae: return Theme.textTertiary
        }
    }
}
