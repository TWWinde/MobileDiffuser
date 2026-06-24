// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI

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

    private enum CanvasState { case generating(Int, Int), loading, result(CGImage), empty }

    private var state: CanvasState {
        if case .generating(let s, let t) = model.phase { return .generating(s, t) }
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
                placeholder(icon: "sparkles", pulsing: true, text: "Generating… step \(s)/\(t)")
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
        // Progress rail renders over the result image too, so progress stays visible.
        .overlay(alignment: .bottom) {
            if case .generating(let s, let t) = model.phase {
                ProgressView(value: Double(s), total: Double(t))
                    .tint(Theme.accent)
                    .padding(Theme.Space.lg)
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
}

/// Prompt entry, circular Generate action, and the size / steps / seed controls.
struct PromptBar: View {
    @Bindable var model: AppModel

    var body: some View {
        VStack(spacing: Theme.Space.md) {
            HStack(alignment: .top, spacing: Theme.Space.md) {
                TextField("Describe an image…", text: $model.prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...4)
                    .foregroundStyle(Theme.textPrimary)
                    .padding(Theme.Space.md)
                    .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous).strokeBorder(Theme.hairline))

                Button { Task { await model.generate() } } label: {
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
                    Segmented(selection: $model.size, options: [512, 768, 1024]) { "\($0)" }
                }
                labeledControl("Steps") {
                    Segmented(selection: $model.steps, options: [4, 8, 16]) { "\($0)" }
                }
                labeledControl("Seed") {
                    SeedField(text: $model.seedText)
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
}
