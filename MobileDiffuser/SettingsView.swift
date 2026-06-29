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
            Text("Match your system, or pin to Light or Dark.")
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
            Text("Downloaded weights live here. Free up space anytime from Manage models.")
                .font(.caption)
                .foregroundStyle(Theme.textTertiary)
        }
    }

    private var deviceSection: some View {
        section("Device", icon: "memorychip") {
            row("Memory",
                ByteCountFormatter.string(fromByteCount: model.device.physicalMemoryBytes, countStyle: .memory))
            row("Available for models",
                ByteCountFormatter.string(fromByteCount: model.device.memoryBudgetBytes, countStyle: .memory))
            Text("How much a model may use is what determines whether it runs here.")
                .font(.caption)
                .foregroundStyle(Theme.textTertiary)
        }
    }

    private var aboutSection: some View {
        section("About", icon: "info.circle") {
            row("Version", appVersion)
            row("Engine", "Pure Swift + MLX")
            Text("Open-weight models, generated on-device. Nothing leaves your device.")
                .font(.caption)
                .foregroundStyle(Theme.textTertiary)
        }
    }

    private var appVersion: String {
        let short = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0"
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String
        return build.map { "\(short) (\($0))" } ?? short
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
