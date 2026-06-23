// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import DiffusionCore

/// The built-in model catalog. Z-Image runs everywhere (downloaded in-app via `ModelDownloader`);
/// FLUX.2 is macOS-only (the `flux-2-swift-mlx` pipeline is monolithic and self-downloads on first
/// run; macOS 14+), so its entry is compiled in only on macOS.
enum Catalog {
    static let zImageTurbo = DiffusionModel(
        id: "z-image-turbo-q4",
        displayName: "Z-Image Turbo (6B)",
        family: .zImage,
        publisher: "Tongyi (Alibaba)",
        summary: "4-bit · 8-step · S3-DiT + Qwen3-4B",
        license: .apache2,
        architecture: ArchitectureSpec(family: .zImage, latentChannels: 16,
            defaultSampler: .flowMatchEuler, defaultSteps: 8, defaultGuidance: 1.0,
            vaeScale: 0.3611, vaeShift: 0.1159),
        variants: [ModelVariant(precision: .q4, approximateBytes: 5_900_000_000,
            components: ComponentSizes(transformer: 3_460_000_000, textEncoder: 2_260_000_000, vae: 160_000_000),
            layout: .mfluxShard,
            source: ModelSource(huggingFaceRepo: "deepsweet/Z-Image-Turbo-6B-MLX-Q4"))])

    #if os(macOS)
    static let flux2Klein = DiffusionModel(
        id: "flux2-klein-4b",
        displayName: "FLUX.2 Klein (4B)",
        family: .flux2,
        publisher: "Black Forest Labs",
        summary: "macOS · selectable precision · Qwen3-4B encoder",
        license: .apache2,
        architecture: ArchitectureSpec(family: .flux2, latentChannels: 16,
            defaultSampler: .flowMatchEuler, defaultSteps: 8, defaultGuidance: 1.0),
        variants: [ModelVariant(precision: .q4, approximateBytes: 4_600_000_000,
            components: ComponentSizes(transformer: 2_180_000_000, textEncoder: 2_260_000_000, vae: 170_000_000),
            layout: .quantoInt,
            source: ModelSource(huggingFaceRepo: "black-forest-labs/FLUX.2-klein-4B"))])
    #endif

    static var all: [DiffusionModel] {
        #if os(macOS)
        [zImageTurbo, flux2Klein]
        #else
        [zImageTurbo]
        #endif
    }
}
