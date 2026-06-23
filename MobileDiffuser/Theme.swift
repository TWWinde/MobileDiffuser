// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI
import DiffusionCore

/// Dark "creative studio" design tokens: near-black surfaces, a single violet generative accent,
/// hairline borders. Shared by every screen.
enum Theme {
    static let accent = Color(red: 0.55, green: 0.36, blue: 0.96)   // violet
    static let bg = Color(white: 0.055)
    static let surface = Color(white: 0.10)
    static let surface2 = Color(white: 0.13)
    static let hairline = Color.white.opacity(0.08)
    static let corner: CGFloat = 16
}

extension View {
    /// A surface card with hairline border.
    func studioCard(_ padding: CGFloat = 14) -> some View {
        self.padding(padding)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: Theme.corner, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: Theme.corner, style: .continuous).strokeBorder(Theme.hairline))
    }
}

/// A small pill (precision chip, tag).
struct Chip: View {
    let text: String
    var filled = false
    var body: some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(filled ? Theme.accent.opacity(0.22) : Theme.surface2,
                        in: Capsule())
            .foregroundStyle(filled ? Theme.accent : .secondary)
    }
}

/// Hardware-fit badge: green = runs resident, amber = two-phase / streams, gray = needs more memory.
struct FitBadge: View {
    let capabilities: EngineCapabilities
    private var color: Color {
        guard capabilities.runnable else { return .gray }
        switch capabilities.residency {
        case .resident: return .green
        case .twoPhase, .streamingInternal, .streamingExternal: return .orange
        case .unsupported: return .gray
        }
    }
    private var label: String {
        guard capabilities.runnable else { return "Needs more memory" }
        switch capabilities.residency {
        case .resident: return "Runs great"
        case .twoPhase: return "Two-phase"
        case .streamingInternal: return "Streams"
        case .streamingExternal: return "Streams (SSD)"
        case .unsupported: return "Unsupported"
        }
    }
    var body: some View {
        Label(label, systemImage: "circle.fill")
            .font(.caption2.weight(.medium))
            .foregroundStyle(color)
            .labelStyle(DotLabelStyle())
    }
}

/// Renders the label icon as a small colored dot before the text.
private struct DotLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 5) {
            configuration.icon.font(.system(size: 7))
            configuration.title
        }
    }
}
