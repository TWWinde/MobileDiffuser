// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI

/// Storage + device info. (External-SSD streaming and per-model location are on the roadmap.)
struct SettingsView: View {
    @Bindable var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                section("Storage") {
                    row("Downloaded models", model.storageLocation, mono: true)
                }
                section("Device") {
                    row("Memory", ByteCountFormatter.string(fromByteCount: model.device.physicalMemoryBytes, countStyle: .memory))
                    row("Working-set budget", ByteCountFormatter.string(fromByteCount: model.device.memoryBudgetBytes, countStyle: .memory))
                    row("Class", model.device.isPhone ? "iPhone / iPad" : "Mac")
                    row("Default precision", model.device.defaultPrecision.label)
                }
                section("About") {
                    row("Engine", "pure Swift + MLX")
                    row("Models", "Z-Image Turbo · FLUX.2 Klein (macOS)")
                    Text("Open-weight, on-device generation. External-SSD streaming and on-disk image library are on the roadmap.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(20)
        }
        .background(Theme.bg)
    }

    private func section(_ title: String, @ViewBuilder _ content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased()).font(.caption2.weight(.semibold)).foregroundStyle(.tertiary)
            VStack(alignment: .leading, spacing: 8) { content() }.studioCard()
        }
    }

    private func row(_ k: String, _ v: String, mono: Bool = false) -> some View {
        HStack(alignment: .top) {
            Text(k).foregroundStyle(.secondary)
            Spacer()
            Text(v).multilineTextAlignment(.trailing).font(mono ? .caption.monospaced() : .subheadline)
                .textSelection(.enabled)
        }.font(.subheadline)
    }
}
