// swift-tools-version: 5.10
import PackageDescription

// AppEngines — a thin wrapper that lets the universal app pull in the FLUX engine. The
// `flux2-diffusion-engine` facade and its flux-2-swift-mlx backend are now cross-platform, so
// `Flux2DiffusionEngine` is depended on for both iOS and macOS (on iPhone it loads the
// pre-quantized 4-bit Klein checkpoint via the two-phase pipeline). Z-Image is depended on directly
// by the app. See docs/BLUEPRINT.md ("FLUX on iOS").
//
// This wrapper still exists for graph hygiene: it keeps the FLUX dependency in one place so the
// app target's dependency list stays simple.
let package = Package(
    name: "AppEngines",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "AppEngines", targets: ["AppEngines"])],
    dependencies: [
        .package(path: "../../flux2-diffusion-engine"),
    ],
    targets: [
        .target(
            name: "AppEngines",
            dependencies: [
                .product(name: "Flux2DiffusionEngine", package: "flux2-diffusion-engine"),
            ]
        ),
    ]
)
