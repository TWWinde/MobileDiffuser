// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI

/// Read-only studio settings: where weights live, what this device can do, how the app looks, and
/// what powers it. No mutable state — the OS drives light/dark, so there is no in-app appearance
/// toggle. Every value is sourced from `AppModel` / `DeviceTier`; no row is invented.
struct SettingsView: View {
    @Bindable var model: AppModel
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.xl) {
                storageSection
                deviceSection
                appearanceSection
                aboutSection
            }
            .frame(maxWidth: 640, alignment: .leading)   // keep readable on wide Mac detail panes
            .frame(maxWidth: .infinity)
            .padding(Theme.Space.xl)
        }
        .background(Theme.bg)
    }

    // MARK: Sections

    private var storageSection: some View {
        section("Storage", icon: "internaldrive") {
            row("Downloaded models", model.storageLocation, mono: true)
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

    private var appearanceSection: some View {
        section("Appearance", icon: "circle.lefthalf.filled") {
            row("Theme", colorScheme == .dark ? "Dark" : "Light")
            Text("MobileDiffuser follows your system appearance. Change it in \(systemSettingsName) to switch themes.")
                .font(.caption)
                .foregroundStyle(Theme.textTertiary)
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

    private var systemSettingsName: String {
        #if os(macOS)
        "System Settings › Appearance"
        #else
        "Settings › Display & Brightness"
        #endif
    }

    // MARK: Building blocks

    private func section(_ title: String, icon: String,
                         @ViewBuilder _ content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.sm) {
            Label {
                Text(title.uppercased())
            } icon: {
                Image(systemName: icon)
            }
            .font(.caption2.weight(.semibold))
            .foregroundStyle(Theme.textTertiary)
            .accessibilityAddTraits(.isHeader)

            VStack(alignment: .leading, spacing: Theme.Space.sm) { content() }
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
