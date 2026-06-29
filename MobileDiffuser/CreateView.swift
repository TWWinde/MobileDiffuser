// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI
import PhotosUI
import ImageIO

/// Generation workspace: a full-bleed result canvas above a prompt + controls panel.
struct CreateView: View {
    @Bindable var model: AppModel
    @State private var showModels = false

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()
            VStack(spacing: 0) {
                modelBar
                HeroCanvas(model: model)
                PromptBar(model: model)
            }
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button { showModels = true } label: {
                    Image(systemName: "gearshape")
                }
                .accessibilityLabel("Manage models")
            }
        }
        .sheet(isPresented: $showModels) { ModelsSheet(model: model) }
    }

    // Two rows so the status + memory readout get the full bar width and never truncate on a narrow
    // iPhone: row 1 is the model identity (name + fit badge + disclosure), row 2 is live status.
    private var modelBar: some View {
        Button { showModels = true } label: {
            VStack(spacing: 4) {
                HStack(spacing: Theme.Space.sm) {
                    Image(systemName: "cube.box.fill").foregroundStyle(Theme.accent)
                    Text(model.selected.displayName)
                        .font(.subheadline.weight(.semibold)).foregroundStyle(Theme.textPrimary)
                        .lineLimit(1)
                    FitBadge(capabilities: model.capabilities(for: model.selected))
                    Spacer(minLength: Theme.Space.sm)
                    Image(systemName: "chevron.right").font(.caption2).foregroundStyle(Theme.textTertiary)
                }
                HStack(spacing: Theme.Space.sm) {
                    Text(model.statusText).font(.caption2)
                        .foregroundStyle(model.isFailed ? Theme.danger : Theme.textSecondary)
                        .lineLimit(1)
                    Spacer(minLength: Theme.Space.sm)
                    if let memory = model.memoryReadout {
                        Text(memory).font(.caption2).monospacedDigit()
                            .foregroundStyle(Theme.textTertiary)
                            .lineLimit(1)
                    }
                }
            }
            .padding(.horizontal, Theme.Space.lg).padding(.vertical, Theme.Space.sm)
            .background(Theme.surface)
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Selected model: \(model.selected.displayName), \(model.statusText). Opens model management.")
        .accessibilityAddTraits(.isButton)
    }
}

/// The result/preview canvas. Renders exactly one of four states, evaluated top-to-bottom
/// (first match wins): generating → loading → result → empty.
struct HeroCanvas: View {
    @Bindable var model: AppModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private enum CanvasState { case generating(Int, Int), pausing(Int, Int), paused(Int, Int), cooling(Int, Int), loading, result(CGImage), empty }

    private var state: CanvasState {
        if case .generating(let s, let t) = model.phase { return .generating(s, t) }
        if case .pausing(let s, let t) = model.phase { return .pausing(s, t) }
        if case .paused(let s, let t) = model.phase { return .paused(s, t) }
        if case .cooling(let s, let t) = model.phase { return .cooling(s, t) }
        if case .downloading = model.phase { return .loading }
        if case .loading = model.phase { return .loading }
        if let cg = model.image { return .result(cg) }
        return .empty
    }

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: Theme.Radius.canvas, style: .continuous).fill(Theme.surface)
            RoundedRectangle(cornerRadius: Theme.Radius.canvas, style: .continuous).strokeBorder(Theme.hairline)

            switch state {
            case .result(let cg):
                Image(decorative: cg, scale: 1).resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.canvas, style: .continuous))
                    .transition(.opacity.combined(with: .scale(scale: 0.98)))
            case .generating(let s, let t):
                forming(icon: "sparkles", pulsing: true, text: "Generating… step \(s)/\(t)")
            case .pausing(let s, let t):
                forming(icon: "pause.circle", pulsing: true, text: "Pausing after step \(s)/\(t)…")
            case .paused(let s, let t):
                forming(icon: "pause.circle.fill", pulsing: false, text: "Paused at step \(s)/\(t)")
            case .cooling(let s, let t):
                forming(icon: "thermometer.snowflake", pulsing: true,
                        text: "Cooling to protect your phone… (step \(s)/\(t))")
            case .loading:
                placeholder(icon: loadingIcon, pulsing: true, text: model.statusText)
            case .empty:
                VStack(spacing: Theme.Space.md) {
                    placeholder(icon: "photo.artframe", pulsing: false, text: "Describe an image, then Generate")
                    if model.isFailed {
                        Text(model.statusText).font(.caption).foregroundStyle(Theme.danger)
                            .multilineTextAlignment(.center).padding(.horizontal, Theme.Space.xl)
                    }
                }
            }
        }
        // Controls float over the canvas. No step progress bar here — the top model bar shows the
        // step and the live latent preview shows the image forming, so a rail on the image is noise.
        .overlay(alignment: .bottom) {
            if generationProgress != nil {
                HStack(spacing: Theme.Space.sm) {
                    Button { model.isGenerationPaused ? model.resumeGeneration() : model.pauseGeneration() } label: {
                        Image(systemName: model.isGenerationPaused ? "play.fill" : "pause.fill")
                            .frame(width: 32, height: 32)
                    }
                    .buttonStyle(.plain)
                    .background(Theme.surface2, in: Circle())
                    .accessibilityLabel(model.isGenerationPaused ? "Resume generation" : "Pause generation")

                    Button { model.cancelOperation() } label: {
                        Image(systemName: "xmark")
                            .frame(width: 32, height: 32)
                    }
                    .buttonStyle(.plain)
                    .background(Theme.surface2, in: Circle())
                    .accessibilityLabel("Cancel generation")
                }
                .padding(Theme.Space.lg)
            } else if model.isBusy, case .downloading(let fraction) = model.phase {
                VStack(spacing: Theme.Space.sm) {
                    ProgressView(value: fraction).tint(Theme.accent)
                    Button { model.cancelOperation() } label: {
                        Image(systemName: "pause.fill")
                            .frame(width: 32, height: 32)
                    }
                    .buttonStyle(.plain)
                    .background(Theme.surface2, in: Circle())
                    .accessibilityLabel("Pause download")
                }
                .padding(Theme.Space.lg)
            } else if model.isBusy, case .loading = model.phase {
                Button { model.cancelOperation() } label: {
                    Image(systemName: "xmark")
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.plain)
                .background(Theme.surface2, in: Circle())
                .padding(Theme.Space.lg)
                .accessibilityLabel("Cancel loading")
            }
        }
        // Dismiss "×" on a finished result — clears the canvas back to empty (the image stays in
        // Library). Lets the user move on to the next prompt without the old result lingering.
        .overlay(alignment: .topTrailing) {
            if case .result = state {
                Button { model.clearResult() } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(.white, .black.opacity(0.35))
                        .padding(Theme.Space.md)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Dismiss result")
            }
        }
        #if os(macOS)
        .frame(maxWidth: 880)
        #endif
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(Theme.Space.lg)
        .animation(Motion.canvas, value: model.image != nil)
    }

    private var loadingIcon: String {
        if case .downloading = model.phase { return "tray.and.arrow.down" }
        return "sparkles"
    }

    private var generationProgress: (step: Int, total: Int)? {
        switch model.phase {
        case .generating(let s, let t), .pausing(let s, let t), .paused(let s, let t), .cooling(let s, let t):
            return (s, t)
        default:
            return nil
        }
    }

    private func placeholder(icon: String, pulsing: Bool, text: String) -> some View {
        VStack(spacing: Theme.Space.md) {
            Image(systemName: icon)
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(Theme.textSecondary)
                .symbolEffect(.pulse, isActive: pulsing && !reduceMotion)
            Text(text).font(.callout).foregroundStyle(Theme.textSecondary)
                .multilineTextAlignment(.center)
        }
    }

    /// During a run: show the cheap latent preview (the image visibly forming) when the architecture
    /// provides one; otherwise the icon placeholder. The step/status text lives in the top model bar,
    /// so no chip on the image; the pause/cancel controls float at the bottom.
    @ViewBuilder
    private func forming(icon: String, pulsing: Bool, text: String) -> some View {
        if let preview = model.previewImage {
            Image(decorative: preview, scale: 1).resizable().aspectRatio(contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.canvas, style: .continuous))
        } else {
            placeholder(icon: icon, pulsing: pulsing, text: text)
        }
    }
}

/// Prompt entry, circular Generate action, and the size / steps / seed controls.
struct PromptBar: View {
    @Bindable var model: AppModel
    @State private var pickerItems: [PhotosPickerItem] = []

    var body: some View {
        VStack(spacing: Theme.Space.md) {
            if model.selected.family == .flux2 { referenceStrip }   // FLUX.2 reference-context i2i
            HStack(alignment: .top, spacing: Theme.Space.md) {
                TextField("Describe an image…", text: $model.prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...4)
                    .foregroundStyle(Theme.textPrimary)
                    .padding(Theme.Space.md)
                    .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous).strokeBorder(Theme.hairline))

                Button { model.startGenerate() } label: {
                    Image(systemName: "arrow.up").font(.headline)
                        .foregroundStyle(Theme.onAccent)
                        .frame(width: 44, height: 44)
                        .background(Theme.accent, in: Circle())
                }
                .buttonStyle(.plain)
                .disabled(model.isBusy || model.prompt.isEmpty)
                .opacity(model.isBusy || model.prompt.isEmpty ? 0.45 : 1)
                .accessibilityLabel("Generate")
                .accessibilityHint("Creates an image from your prompt")
            }
            HStack(alignment: .bottom, spacing: Theme.Space.lg) {
                labeledControl("Size") {
                    Segmented(selection: $model.size, options: model.selected.sizeChoices) { "\($0)" }
                        .disabled(model.isBusy)
                }
                labeledControl("Steps") {
                    Segmented(selection: $model.steps, options: model.selected.stepChoices) { "\($0)" }
                        .disabled(model.isBusy)
                }
                labeledControl("Seed") {
                    SeedField(text: $model.seedText)
                        .disabled(model.isBusy)
                }
                .frame(maxWidth: 120)
            }
        }
        // Compose the workspace on wide Macs: cap content to the canvas width, centered,
        // while the surface bar still spans edge to edge.
        .frame(maxWidth: 880)
        .padding(Theme.Space.lg)
        .frame(maxWidth: .infinity)
        .background(Theme.surface)
        .overlay(alignment: .top) { Divider().background(Theme.hairline) }
    }

    /// A control with a small caption above it (Size / Steps / Seed).
    @ViewBuilder private func labeledControl<C: View>(_ title: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption2).foregroundStyle(Theme.textTertiary)
            content()
        }
    }

    /// FLUX.2 reference-context i2i: pick 1–3 reference images (editing / style / composition). Adding
    /// any reference routes the run to the resident facade (the streaming path is text-to-image only),
    /// so on iPhone i2i runs at 512.
    @ViewBuilder private var referenceStrip: some View {
        HStack(spacing: Theme.Space.sm) {
            ForEach(Array(model.referenceImages.enumerated()), id: \.offset) { idx, cg in
                Image(decorative: cg, scale: 1).resizable().aspectRatio(contentMode: .fill)
                    .frame(width: 44, height: 44)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(Theme.hairline))
                    .overlay(alignment: .topTrailing) {
                        Button {
                            model.referenceImages.remove(at: idx); pickerItems.removeAll()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .symbolRenderingMode(.palette).foregroundStyle(.white, .black.opacity(0.5))
                        }
                        .buttonStyle(.plain).offset(x: 5, y: -5)
                    }
            }
            if model.referenceImages.count < 3 {
                PhotosPicker(selection: $pickerItems, maxSelectionCount: 3, matching: .images) {
                    VStack(spacing: 2) {
                        Image(systemName: "photo.badge.plus")
                        Text(model.referenceImages.isEmpty ? "Reference" : "Add").font(.caption2)
                    }
                    .frame(width: 44, height: 44).foregroundStyle(Theme.textSecondary)
                    .background(Theme.surface2, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(Theme.hairline))
                }
                .disabled(model.isBusy)
            }
            Spacer(minLength: 0)
            if !model.referenceImages.isEmpty {
                Text("Edit / reference").font(.caption2).foregroundStyle(Theme.textTertiary)
            }
        }
        .onChange(of: pickerItems) { _, items in Task { await loadReferences(items) } }
    }

    private func loadReferences(_ items: [PhotosPickerItem]) async {
        var images: [CGImage] = []
        for item in items.prefix(3) {
            if let data = try? await item.loadTransferable(type: Data.self),
               let src = CGImageSourceCreateWithData(data as CFData, nil),
               let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) {
                images.append(cg)
            }
        }
        await MainActor.run { model.referenceImages = images }
    }
}
