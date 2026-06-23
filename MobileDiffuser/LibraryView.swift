// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI

/// Your generated images. Tap one to see its prompt/params and reuse the settings.
/// (In-session for now; on-disk persistence is a follow-up.)
struct LibraryView: View {
    @Bindable var model: AppModel
    @State private var selected: Generation?

    private let columns = [GridItem(.adaptive(minimum: 120), spacing: 10)]

    var body: some View {
        Group {
            if model.history.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "photo.on.rectangle.angled")
                        .font(.system(size: 40, weight: .light)).foregroundStyle(.secondary)
                    Text("Your generations appear here").font(.callout).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 10) {
                        ForEach(model.history) { gen in
                            Image(decorative: gen.image, scale: 1).resizable().aspectRatio(1, contentMode: .fill)
                                .frame(maxWidth: .infinity).clipped()
                                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                                .contentShape(Rectangle())
                                .onTapGesture { selected = gen }
                        }
                    }.padding(16)
                }
            }
        }
        .background(Theme.bg)
        .sheet(item: $selected) { gen in GenerationDetail(model: model, gen: gen) }
    }
}

private struct GenerationDetail: View {
    @Bindable var model: AppModel
    let gen: Generation
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Image(decorative: gen.image, scale: 1).resizable().aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                Text(gen.prompt).font(.callout)
                VStack(alignment: .leading, spacing: 8) {
                    row("Model", gen.modelName)
                    row("Size", "\(gen.size)×\(gen.size)")
                    row("Steps", "\(gen.steps)")
                    row("Seed", "\(gen.seed)")
                }.studioCard()
                Button { model.reuse(gen); dismiss() } label: {
                    Label("Reuse settings", systemImage: "arrow.uturn.left").frame(maxWidth: .infinity)
                }.buttonStyle(.borderedProminent).tint(Theme.accent)
            }
            .padding(20)
        }
        .background(Theme.bg)
        .preferredColorScheme(.dark)
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack { Text(k).foregroundStyle(.secondary); Spacer(); Text(v).monospacedDigit() }.font(.subheadline)
    }
}
