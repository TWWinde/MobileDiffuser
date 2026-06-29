// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import Foundation

/// A unified, UI-facing description of one model's installable pieces and optional precision
/// choices. Every model — single-file (Z-Image) or multi-component with precision (FLUX) — is
/// described as a `ModelRecipe`, so the Details screen renders one data-driven template instead of
/// per-family layouts. The app builds these from each engine's specifics (see `AppModel.recipe`).
struct RecipeComponent: Identifiable, Sendable {
    enum Kind: String, Sendable {
        case transformer = "Transformer"
        case textEncoder = "Text encoder"
        case vae = "VAE"
        case weights = "Weights"
    }
    let id: String
    let title: String
    let subtitle: String
    let kind: Kind
    let repo: String
    let bytes: Int64
    let isDownloaded: Bool
    /// Whether this component is part of the currently selected precision recipe (what generation runs).
    let isActive: Bool
}

struct PrecisionOption: Identifiable, Sendable {
    let id: String
    let label: String
    let note: String
}

/// One precision dimension (e.g. transformer precision, text-encoder precision). Empty for models
/// with a single fixed precision.
struct PrecisionAxis: Identifiable, Sendable {
    let id: String
    let title: String
    let options: [PrecisionOption]
    let selectedID: String

    var selectedOption: PrecisionOption? { options.first { $0.id == selectedID } }
}

struct ModelRecipe: Sendable {
    var axes: [PrecisionAxis] = []
    var components: [RecipeComponent] = []

    /// Active components not yet on disk.
    var missing: [RecipeComponent] { components.filter { $0.isActive && !$0.isDownloaded } }
    /// The active recipe is fully installed.
    var isInstalled: Bool { missing.isEmpty }
    var missingBytes: Int64 { missing.reduce(0) { $0 + $1.bytes } }
    /// Total download size of the active recipe (installed or not).
    var activeBytes: Int64 { components.filter(\.isActive).reduce(0) { $0 + $1.bytes } }
    var activeCount: Int { components.filter(\.isActive).count }
    /// Total bytes of components currently on disk.
    var bytesOnDisk: Int64 { components.filter(\.isDownloaded).reduce(0) { $0 + $1.bytes } }
}
