// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI
import UniformTypeIdentifiers

/// Your generated images. The grid shows every saved local generation (newest first); tapping a
/// thumbnail opens a detail sheet with its prompt, params, a one-tap "Reuse settings", and an
/// export action (Save to Photos on iOS / Export PNG on macOS).
struct LibraryView: View {
    @Bindable var model: AppModel
    @State private var selected: Generation?

    private let columns = [GridItem(.adaptive(minimum: 120), spacing: Theme.Space.sm)]

    var body: some View {
        Group {
            if model.history.isEmpty {
                emptyState
            } else {
                grid
            }
        }
        .background(Theme.bg)
        .sheet(item: $selected) { gen in
            GenerationDetail(model: model, gen: gen)
                #if os(macOS)
                .frame(minWidth: 660, idealWidth: 860, minHeight: 460, idealHeight: 600)
                #endif
        }
    }

    // MARK: Empty state

    private var emptyState: some View {
        VStack(spacing: Theme.Space.md) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 40, weight: .light))
                .foregroundStyle(Theme.textSecondary)
                .accessibilityHidden(true)
            Text("Your generations appear here")
                .font(.callout)
                .foregroundStyle(Theme.textSecondary)
                .multilineTextAlignment(.center)
        }
        .padding(Theme.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // The text alone conveys the state to VoiceOver; the icon is decorative.
    }

    // MARK: Grid

    private var grid: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: Theme.Space.sm) {
                ForEach(model.history) { gen in
                    Thumbnail(gen: gen) { selected = gen }
                }
            }
            .padding(Theme.Space.lg)
        }
    }
}

// MARK: - Thumbnail cell

/// A single square library cell. The image is decorative (hidden from VoiceOver); the cell itself
/// is the accessible, tappable button carrying a truncated prompt as its label.
private struct Thumbnail: View {
    let gen: Generation
    let onTap: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var pressed = false

    var body: some View {
        Image(decorative: gen.image, scale: 1)
            .resizable()
            .aspectRatio(1, contentMode: .fill)
            .frame(maxWidth: .infinity)
            .clipped()
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.field, style: .continuous)
                    .strokeBorder(Theme.hairline)
            )
            .scaleEffect(pressed && !reduceMotion ? 0.97 : 1)
            .animation(Motion.press, value: pressed)
            .contentShape(Rectangle())
            // A manual press gesture gives the tactile scale while keeping a single tap action.
            .onTapGesture { onTap() }
            ._onPressGesture { pressed = $0 }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(label)
            .accessibilityHint("Show details")
            .accessibilityAddTraits(.isButton)
    }

    private var label: String {
        let trimmed = gen.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "Generated image" : String(trimmed.prefix(80))
    }
}

private extension View {
    /// Reports press state (down → `true`, up/cancel → `false`) without consuming the tap.
    func _onPressGesture(_ action: @escaping (Bool) -> Void) -> some View {
        simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in action(true) }
                .onEnded { _ in action(false) }
        )
    }
}

// MARK: - Generation detail

/// The detail sheet for one generation: full image, prompt, a parameter table, "Reuse settings",
/// and an export action. Reads/uses `AppModel` (reuse + export) but adds no model state.
private struct GenerationDetail: View {
    @Bindable var model: AppModel
    let gen: Generation
    @Environment(\.dismiss) private var dismiss
    @State private var confirmDelete = false

    #if os(macOS)
    @State private var exporting = false
    #endif

    /// Temp .png (with embedded metadata) backing the share sheet. Written once when the detail
    /// appears and removed on disappear, so re-renders don't spawn extra temp files.
    @State private var shareURL: URL?

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, Theme.Space.xl)
                .padding(.top, Theme.Space.lg)
                .padding(.bottom, Theme.Space.md)
            Divider().background(Theme.hairline)
            content
        }
        .background(Theme.bg)
        .onAppear {
            if let data = model.pngData(for: gen) { shareURL = model.writeTempPNG(data, for: gen) }
        }
        .onDisappear {
            if let url = shareURL { try? FileManager.default.removeItem(at: url) }
            shareURL = nil
        }
        // Confirmation banner after Save to Photos / Export — shared with Create via `toastBanner`.
        .toastBanner(model.toast)
        .alert("Delete this generation?", isPresented: $confirmDelete) {
            Button("Delete", role: .destructive) {
                model.deleteGeneration(gen)
                dismiss()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes the saved PNG from the local Library.")
        }
    }

    #if os(macOS)
    /// Wide layout: image on the left, info on the right — everything fits without scrolling.
    private var content: some View {
        HStack(alignment: .top, spacing: Theme.Space.xl) {
            imageView
            infoColumn.frame(width: 300)
        }
        .padding(Theme.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }
    #else
    /// Tall layout: cap the image so the prompt, details, and actions stay in view.
    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.lg) {
                imageView.frame(maxHeight: 360)
                infoColumn
            }
            .padding(Theme.Space.xl)
        }
    }
    #endif

    private var imageView: some View {
        Image(decorative: gen.image, scale: 1)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(maxWidth: .infinity)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.card, style: .continuous)
                    .strokeBorder(Theme.hairline)
            )
            .accessibilityLabel("Generated image")
    }

    private var infoColumn: some View {
        VStack(alignment: .leading, spacing: Theme.Space.lg) {
            prompt
            params
            actions
        }
    }

    // MARK: Header (title + close)

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Generation")
                .font(.title2.weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
            Spacer(minLength: Theme.Space.md)
            Button { dismiss() } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)
                    .foregroundStyle(Theme.textSecondary)
                    .symbolRenderingMode(.hierarchical)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Close")
        }
    }

    // MARK: Prompt

    private var prompt: some View {
        VStack(alignment: .leading, spacing: Theme.Space.xs) {
            sectionHeader("PROMPT")
            Text(gen.prompt)
                .font(.callout)
                .foregroundStyle(Theme.textPrimary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: Parameters

    private var params: some View {
        VStack(alignment: .leading, spacing: Theme.Space.xs) {
            sectionHeader("DETAILS")
            VStack(alignment: .leading, spacing: Theme.Space.sm) {
                row("Model", gen.modelName)
                // Model-specific recipe (FLUX: transformer/encoder/decoder; Z-Image: precision; …).
                ForEach(gen.settings) { row($0.label, $0.value) }
                row("Size", "\(gen.size)×\(gen.size)")
                row("Steps", "\(gen.steps)")
                row("Seed", "\(gen.seed)")
                row("Time", formatDuration(gen.duration))
                row("Created", gen.date.formatted(date: .abbreviated, time: .shortened))
            }
            .studioCard()
        }
    }

    // MARK: Actions

    private var actions: some View {
        VStack(spacing: Theme.Space.sm) {
            Button {
                model.reuse(gen)
                dismiss()
            } label: {
                Label("Reuse settings", systemImage: "arrow.uturn.left")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle(.primary))
            .accessibilityHint("Loads this prompt and settings into Create")

            saveButton

            shareButton

            Button(role: .destructive) { confirmDelete = true } label: {
                Label("Delete", systemImage: "trash")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle(.secondary))
        }
    }

    @ViewBuilder private var saveButton: some View {
        #if os(iOS)
        Button { model.exportImage(gen) } label: {
            Label("Save to Photos", systemImage: "square.and.arrow.down")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(StudioButtonStyle(.secondary))
        .accessibilityLabel("Save to Photos")
        .accessibilityHint("Adds this image to your photo library")
        #elseif os(macOS)
        Button { exporting = true } label: {
            Label("Export image…", systemImage: "square.and.arrow.down")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(StudioButtonStyle(.secondary))
        .accessibilityLabel("Export image")
        .accessibilityHint("Saves this image as a PNG file")
        .fileExporter(
            // Metadata-bearing PNG (prompt/seed/model embedded as text chunks).
            isPresented: $exporting,
            document: PNGDocument(data: model.pngData(for: gen)),
            contentType: .png,
            defaultFilename: model.exportBasename(for: gen)
        ) { result in
            if case .success = result { model.showToast("Image exported") }
        }
        #endif
    }

    /// System share sheet (AirDrop / Messages / Save to Files …). Shares a metadata-bearing PNG written
    /// to a temp .png URL — a file URL shares more reliably across targets and carries a proper filename.
    /// Cross-platform: `ShareLink` works on both iOS and macOS.
    @ViewBuilder private var shareButton: some View {
        if let url = shareURL {
            ShareLink(item: url, preview: SharePreview(shareTitle, image: Image(decorative: gen.image, scale: 1))) {
                Label("Share…", systemImage: "square.and.arrow.up")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle(.secondary))
            .accessibilityLabel("Share image")
            .accessibilityHint("Opens the system share sheet")
        }
    }

    /// Short share-preview title: a trimmed prompt, or a stable fallback.
    private var shareTitle: String {
        let trimmed = gen.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "Generated image" : String(trimmed.prefix(60))
    }

    // MARK: Pieces

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(Theme.textTertiary)
            .accessibilityHidden(true)
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(k).foregroundStyle(Theme.textSecondary)
            Spacer(minLength: Theme.Space.md)
            Text(v)
                .foregroundStyle(Theme.textPrimary)
                .monospacedDigit()
                .multilineTextAlignment(.trailing)
        }
        .font(.subheadline)
        .accessibilityElement(children: .combine)
    }
}

#if os(macOS)
/// Lightweight PNG wrapper for `.fileExporter`.
private struct PNGDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.png] }
    var data: Data
    init(data: Data?) { self.data = data ?? Data() }
    init(configuration: ReadConfiguration) throws {
        data = configuration.file.regularFileContents ?? Data()
    }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
#endif
