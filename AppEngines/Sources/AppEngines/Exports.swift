// Re-export the FLUX engine so the app gets `Flux2FacadeEngine` via `import AppEngines`. The facade
// and its flux-2-swift-mlx backend are now cross-platform, so this is exported on both iOS and
// macOS (on iPhone FLUX runs the two-phase pipeline with the pre-quantized 4-bit Klein checkpoint).
// See docs/BLUEPRINT.md ("FLUX on iOS").
@_exported import Flux2DiffusionEngine
