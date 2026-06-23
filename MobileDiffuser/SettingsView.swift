// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI

/// Studio settings: manage models, pick the appearance, see storage + device info, and what powers
/// the app. Models, appearance, and storage usage are interactive; device + about are informational.
struct SettingsView: View {
    @Bindable var model: AppModel
    @State private var showModels = false
    @State private var usedBytes: Int64 = 0

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.xl) {
                modelsSection
                appearanceSection
                storageSection
                deviceSection
                aboutSection
            }
            .frame(maxWidth: 640, alignment: .leading)   // keep readable on wide Mac detail panes
            .frame(maxWidth: .infinity)
            .padding(Theme.Space.xl)
        }
        .background(Theme.bg)
        .task(id: showModels) { usedBytes = await model.storageUsedBytes() }
        .sheet(isPresented: $showModels) { ModelsSheet(model: model) }
    }

    // MARK: Sections

    private var modelsSection: some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            sectionLabel("Models", icon: "square.stack.3d.up")
            Button { showModels = true } label: {
                HStack(spacing: Theme.Space.md) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Manage models")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(Theme.textPrimary)
                        Text("\(model.downloadedModelCount) of \(model.models.count) installed")
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                    }
                    Spacer(minLength: Theme.Space.md)
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.textTertiary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .studioCard()
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityHint("Download, switch, and inspect models")
        }
    }

    private var appearanceSection: some View {
        section("Appearance", icon: "circle.lefthalf.filled") {
            Segmented(selection: $model.appearance, options: AppTheme.allCases) { $0.label }
                .accessibilityLabel("Theme")
            Text("System follows your device’s light/dark setting.")
                .font(.caption)
                .foregroundStyle(Theme.textTertiary)
        }
    }

    private var storageSection: some View {
        section("Storage", icon: "internaldrive") {
            row("Models on disk", ByteCountFormatter.string(fromByteCount: usedBytes, countStyle: .file))
            VStack(alignment: .leading, spacing: 4) {
                Text("Location").font(.subheadline).foregroundStyle(Theme.textSecondary)
                Text(model.storageLocation)
                    .font(.caption.monospaced()).foregroundStyle(Theme.textTertiary)
                    .lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Location")
            .accessibilityValue(model.storageLocation)
            Text("Z-Image weights download here on first use. FLUX.2 manages its own weights inside the engine on macOS.")
                .font(.caption)
                .foregroundStyle(Theme.textTertiary)
        }
    }

    private var deviceSection: some View {
        section("Device & memory", icon: "memorychip") {
            row("Memory",
                ByteCountFormatter.string(fromByteCount: model.device.physicalMemoryBytes, countStyle: .memory))
            row("Working-set budget",
                ByteCountFormatter.string(fromByteCount: model.device.memoryBudgetBytes, countStyle: .memory))
            row("Class", model.device.isPhone ? "iPhone / iPad" : "Mac")
            row("Default precision", model.device.defaultPrecision.label)
        }
    }

    private var aboutSection: some View {
        section("About", icon: "info.circle") {
            row("Engine", "Pure Swift + MLX")
            row("Models", "Z-Image Turbo · FLUX.2 Klein (macOS)")
            Text("Open-weight, on-device generation. External-SSD streaming and an on-disk image library are on the roadmap.")
                .font(.caption)
                .foregroundStyle(Theme.textTertiary)
        }
    }

    // MARK: Building blocks

    private func sectionLabel(_ title: String, icon: String) -> some View {
        Label {
            Text(title.uppercased())
        } icon: {
            Image(systemName: icon)
        }
        .font(.caption2.weight(.semibold))
        .foregroundStyle(Theme.textTertiary)
        .accessibilityAddTraits(.isHeader)
    }

    private func section(_ title: String, icon: String,
                         @ViewBuilder _ content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            sectionLabel(title, icon: icon)
            VStack(alignment: .leading, spacing: Theme.Space.md) { content() }
                .frame(maxWidth: .infinity, alignment: .leading)
                .studioCard()
        }
    }

    private func row(_ key: String, _ value: String, mono: Bool = false) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: Theme.Space.md) {
            Text(key)
                .font(.subheadline)
                .foregroundStyle(Theme.textSecondary)
            Spacer(minLength: Theme.Space.md)
            Text(value)
                .font(mono ? .caption.monospaced() : .subheadline)
                .foregroundStyle(Theme.textPrimary)
                .multilineTextAlignment(.trailing)
                .textSelection(.enabled)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(key)
        .accessibilityValue(value)
    }
}
