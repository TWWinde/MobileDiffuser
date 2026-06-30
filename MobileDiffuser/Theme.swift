// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI
import DiffusionCore

#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

/// "Creative studio" design tokens: layered flat surfaces, a single violet generative accent,
/// hairline borders. Adaptive to system light/dark (no asset catalog — colors resolve per scheme
/// at draw time). Shared by every screen.
enum Theme {

    // MARK: Dynamic-color shim

    /// Solid dynamic color. `hex` resolves per scheme (`0xRRGGBB`).
    static func dynamic(dark: UInt32, light: UInt32) -> Color {
        #if os(iOS)
        Color(UIColor { $0.userInterfaceStyle == .dark ? UIColor(hex: dark) : UIColor(hex: light) })
        #elseif os(macOS)
        Color(NSColor(name: nil) {
            $0.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? NSColor(hex: dark) : NSColor(hex: light)
        })
        #else
        Color(red: Double((dark >> 16) & 0xFF) / 255, green: Double((dark >> 8) & 0xFF) / 255, blue: Double(dark & 0xFF) / 255)
        #endif
    }

    /// Dynamic color with per-scheme opacity (e.g. `accentSoft`).
    static func dynamic(dark: UInt32, darkA: Double, light: UInt32, lightA: Double) -> Color {
        #if os(iOS)
        Color(UIColor { $0.userInterfaceStyle == .dark ? UIColor(hex: dark, alpha: darkA) : UIColor(hex: light, alpha: lightA) })
        #elseif os(macOS)
        Color(NSColor(name: nil) {
            $0.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? NSColor(hex: dark, alpha: darkA) : NSColor(hex: light, alpha: lightA)
        })
        #else
        Color(red: Double((dark >> 16) & 0xFF) / 255, green: Double((dark >> 8) & 0xFF) / 255, blue: Double(dark & 0xFF) / 255).opacity(darkA)
        #endif
    }

    /// Overload for white/black-based opacity tokens (hairline dark uses pure white).
    static func dynamic(dark: Color, darkA: Double, light: UInt32, lightA: Double) -> Color {
        #if os(iOS)
        let darkUI = UIColor(dark)
        return Color(UIColor { $0.userInterfaceStyle == .dark ? darkUI.withAlphaComponent(darkA) : UIColor(hex: light, alpha: lightA) })
        #elseif os(macOS)
        let darkNS = NSColor(dark)
        return Color(NSColor(name: nil) {
            $0.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? darkNS.withAlphaComponent(darkA) : NSColor(hex: light, alpha: lightA)
        })
        #else
        return dark.opacity(darkA)
        #endif
    }

    // MARK: Color tokens (light + dark)

    /// App background.
    static let bg          = dynamic(dark: 0x0E0E10, light: 0xF6F6F8)
    /// Card / panel / modelBar / canvas fill / selected segment tile.
    static let surface     = dynamic(dark: 0x1A1A1D, light: 0xFFFFFF)
    /// Inset fields, segmented track, chip fill.
    static let surface2    = dynamic(dark: 0x222227, light: 0xECECF0)
    /// All 1px borders.
    static let hairline    = dynamic(dark: .white, darkA: 0.08, light: 0x11111A, lightA: 0.10)

    /// Titles, prompt text, values.
    static let textPrimary   = dynamic(dark: 0xF2F2F5, light: 0x16161A)
    /// Body, captions, labels, keys.
    static let textSecondary = dynamic(dark: 0xA8A8B2, light: 0x5A5A66)
    /// Section headers, chevrons, hints.
    static let textTertiary  = dynamic(dark: 0x6E6E78, light: 0x9A9AA6)

    /// Violet brand / generative accent.
    static let accent     = dynamic(dark: 0x8C5CF5, light: 0x7A45F0)
    /// Filled-chip / glow wash.
    static let accentSoft = dynamic(dark: 0x8C5CF5, darkA: 0.22, light: 0x7A45F0, lightA: 0.14)
    /// Text / icon on a filled accent surface.
    static let onAccent   = Color.white

    /// Fit badge — resident.
    static let fitGreen = dynamic(dark: 0x3DD68C, light: 0x1FA968)
    /// Fit badge — two-phase / streams.
    static let fitAmber = dynamic(dark: 0xF0A33D, light: 0xC9791A)
    /// Fit badge — needs more / unsupported.
    static let fitGray  = dynamic(dark: 0x6E6E78, light: 0x9A9AA6)

    /// Failed-state text.
    static let danger = dynamic(dark: 0xFF5C5C, light: 0xD92D2D)

    // MARK: Scale tokens

    enum Radius {
        static let card: CGFloat = 16     // studioCard, model cards, tables (== Theme.corner)
        static let canvas: CGFloat = 20   // hero canvas
        static let field: CGFloat = 12    // prompt field, library thumbnails, seed field
        static let control: CGFloat = 10  // segmented control, StudioButton
    }

    enum Space {
        static let xs: CGFloat = 6
        static let sm: CGFloat = 10
        static let md: CGFloat = 12
        static let lg: CGFloat = 16
        static let xl: CGFloat = 20
        static let xxl: CGFloat = 28
    }

    /// KEPT for source compat (studioCard, card overlay strokes). Equals `Radius.card`.
    static let corner: CGFloat = 16
}

// MARK: - Hex helpers

#if os(iOS)
private extension UIColor {
    convenience init(hex: UInt32, alpha: Double = 1) {
        self.init(red:   CGFloat((hex >> 16) & 0xFF) / 255,
                  green: CGFloat((hex >> 8) & 0xFF) / 255,
                  blue:  CGFloat(hex & 0xFF) / 255,
                  alpha: CGFloat(alpha))
    }
}
#elseif os(macOS)
private extension NSColor {
    convenience init(hex: UInt32, alpha: Double = 1) {
        self.init(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
                  green:   CGFloat((hex >> 8) & 0xFF) / 255,
                  blue:    CGFloat(hex & 0xFF) / 255,
                  alpha:   CGFloat(alpha))
    }
}
#endif

// MARK: - Card modifier

extension View {
    /// A surface card with hairline border.
    func studioCard(_ padding: CGFloat = 14) -> some View {
        self.padding(padding)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: Theme.corner, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: Theme.corner, style: .continuous).strokeBorder(Theme.hairline))
    }
}

// MARK: - Motion

/// Animation tokens. All `withAnimation` in the app routes through these so Reduce Motion
/// collapses every spring to a short ease.
enum Motion {
    /// System Reduce-Motion flag (for animation selection).
    @MainActor static var reduce: Bool {
        #if os(iOS)
        UIAccessibility.isReduceMotionEnabled
        #elseif os(macOS)
        NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        #else
        false
        #endif
    }
    @MainActor static var spring: Animation { reduce ? .easeInOut(duration: 0.12) : .spring(response: 0.32, dampingFraction: 0.82) }
    @MainActor static var select: Animation { reduce ? .easeInOut(duration: 0.10) : .spring(response: 0.28, dampingFraction: 0.85) }
    @MainActor static var canvas: Animation { reduce ? .easeInOut(duration: 0.15) : .spring(response: 0.40, dampingFraction: 0.90) }
    static var press: Animation { .easeOut(duration: 0.12) }
}

// MARK: - Chip

/// A small pill (precision chip, tag).
struct Chip: View {
    let text: String
    var filled = false
    var body: some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .lineLimit(1).fixedSize()   // a chip is a tag — keep it one line at its intrinsic width
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(filled ? Theme.accentSoft : Theme.surface2, in: Capsule())
            .foregroundStyle(filled ? Theme.accent : Theme.textSecondary)
    }
}

// MARK: - FitBadge

/// Hardware-fit badge: green = runs resident, amber = two-phase / streams, gray = needs more memory.
/// Gated on `runnable` first, then `residency` (which has no associated values). The text label
/// doubles as the accessibility label, so the meaning never relies on color alone.
struct FitBadge: View {
    let capabilities: EngineCapabilities
    private var color: Color {
        guard capabilities.runnable else { return Theme.fitGray }
        switch capabilities.residency {
        case .resident: return Theme.fitGreen
        case .twoPhase, .streamingInternal, .streamingExternal: return Theme.fitAmber
        case .unsupported: return Theme.fitGray
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
            // Keep the badge to one line at its intrinsic width so it never wraps or steals width from a
            // sibling title in a tight row (the model name beside it shrinks instead).
            configuration.title.lineLimit(1).fixedSize()
        }
    }
}

// MARK: - Segmented control

/// A studio-styled segmented control (replaces system `Picker` menus). The selected option is a
/// raised surface tile that slides between segments via `matchedGeometryEffect`.
struct Segmented<T: Hashable>: View {
    @Binding var selection: T
    let options: [T]
    let label: (T) -> String
    @Namespace private var ns

    init(selection: Binding<T>, options: [T], label: @escaping (T) -> String) {
        self._selection = selection
        self.options = options
        self.label = label
    }

    var body: some View {
        HStack(spacing: 0) {
            ForEach(options, id: \.self) { opt in
                let isSelected = opt == selection
                Text(label(opt))
                    .font(.subheadline.weight(isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? Theme.textPrimary : Theme.textSecondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 7)
                    .background {
                        if isSelected {
                            RoundedRectangle(cornerRadius: Theme.Radius.control - 2, style: .continuous)
                                .fill(Theme.surface)
                                .shadow(color: .black.opacity(0.18), radius: 3, y: 1)
                                .matchedGeometryEffect(id: "seg", in: ns)
                        }
                    }
                    .contentShape(Rectangle())
                    .onTapGesture { withAnimation(Motion.spring) { selection = opt } }
            }
        }
        .padding(3)
        .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
    }
}

// MARK: - StudioButtonStyle

/// Rectangular studio buttons. The circular Generate button does NOT use this style.
struct StudioButtonStyle: ButtonStyle {
    enum Kind { case primary, secondary }
    let kind: Kind
    init(_ kind: Kind) { self.kind = kind }

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(kind == .primary ? Theme.onAccent : Theme.textPrimary)
            .padding(.horizontal, Theme.Space.lg)
            .padding(.vertical, Theme.Space.sm)
            .background {
                let shape = RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                ZStack {
                    shape.fill(kind == .primary ? Theme.accent : Theme.surface)
                    if kind == .secondary { shape.strokeBorder(Theme.hairline) }
                }
            }
            .opacity(configuration.isPressed ? 0.82 : 1)
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
            .animation(Motion.press, value: configuration.isPressed)
    }
}

// MARK: - SeedField

/// A themed inset numeric field replacing the system `.roundedBorder` Seed field.
/// Flexible width so it never crops at large Dynamic Type.
struct SeedField: View {
    @Binding var text: String
    var body: some View {
        TextField("Seed", text: $text)
            .textFieldStyle(.plain)
            .foregroundStyle(Theme.textPrimary)
            .multilineTextAlignment(.center)
            .frame(minWidth: 64, idealWidth: 72)
            .padding(.vertical, 7).padding(.horizontal, 10)
            .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous).strokeBorder(Theme.hairline))
            #if os(iOS)
            .keyboardType(.numberPad)
            #endif
    }
}

// MARK: - Toast banner

extension View {
    /// Bottom confirmation toast (e.g. "Saved to Photos", "Cancelled"), shared by Create and Library
    /// so the user sees feedback on whichever screen they're on — not only in the Library.
    func toastBanner(_ toast: String?) -> some View {
        overlay(alignment: .bottom) {
            if let toast {
                Label(toast, systemImage: "checkmark.circle.fill")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(Theme.textPrimary)
                    .padding(.horizontal, Theme.Space.lg).padding(.vertical, Theme.Space.md)
                    .background(Theme.surface, in: Capsule())
                    .overlay(Capsule().strokeBorder(Theme.hairline))
                    .shadow(color: .black.opacity(0.15), radius: 12, y: 4)
                    .padding(.bottom, Theme.Space.xl)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(Motion.canvas, value: toast)
    }
}

// MARK: - ComponentBar (optional enrichment)

/// A proportional bar visualizing a variant's transformer / text-encoder / VAE split, with a
/// 3-item legend. Purely a presentation of existing `ComponentSizes` — invents no model state.
struct ComponentBar: View {
    let components: ComponentSizes

    private var total: Double {
        Double(components.transformer + components.textEncoder + components.vae)
    }
    private func fraction(_ part: Int64) -> Double {
        total > 0 ? Double(part) / total : 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            GeometryReader { geo in
                HStack(spacing: 0) {
                    Rectangle().fill(Theme.accent)
                        .frame(width: geo.size.width * fraction(components.transformer))
                    Rectangle().fill(Theme.accent.opacity(0.55))
                        .frame(width: geo.size.width * fraction(components.textEncoder))
                    Rectangle().fill(Theme.textTertiary)
                        .frame(width: geo.size.width * fraction(components.vae))
                }
            }
            .frame(height: 6)
            .clipShape(Capsule())

            HStack(spacing: 12) {
                legend("Transformer", Theme.accent)
                legend("Text encoder", Theme.accent.opacity(0.55))
                legend("VAE", Theme.textTertiary)
            }
        }
    }

    private func legend(_ text: String, _ color: Color) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(text).font(.caption2).foregroundStyle(Theme.textSecondary)
        }
    }
}
