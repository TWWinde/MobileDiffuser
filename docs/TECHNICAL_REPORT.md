# Technical Report

MobileDiffuser is a practical experiment in deploying distilled SD3 Medium on
iPhone with Core ML. The project focuses on a reproducible 512 x 512 path and
documents the conversion/runtime tradeoffs that mattered most.

For current implementation details, read:

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [REPRODUCING_MODELS.md](REPRODUCING_MODELS.md)
- [IPHONE_OOM_DEBUG.md](IPHONE_OOM_DEBUG.md)

## Goals

1. Run image generation locally on iPhone.
2. Prefer Apple Neural Engine instead of CPU fallback.
3. Avoid app termination from memory pressure.
4. Keep the app usable for comparing two-step and four-step distilled models.
5. Make the conversion process repeatable without committing model weights.

## Main Findings

### Single huge MMDiT is fragile on device

The SD3 Medium transformer is large enough that a single Core ML program can
fail ANE compilation or create unacceptable plan-build memory pressure.

Splitting the transformer into stages makes each Core ML compilation unit
smaller and easier for the device to load.

### Weight-only INT8 compression is useful

INT8 linear symmetric weight quantization cuts MMDiT stage weight size without
requiring a full custom quantized runtime. Core ML still controls activation
precision and op placement.

This is a pragmatic compromise: smaller model files and lower load pressure,
while preserving the standard Core ML execution path.

### CPU fallback hides real deployment failures

If CPU fallback is enabled during validation, the app may appear to work while
generation becomes unusably slow. MobileDiffuser currently keeps the normal
path ANE-first so ANE failures remain visible.

### Prewarm can make memory behavior worse

Eagerly compiling every submodel before generation can create a large initial
memory spike. Lazy first-generation loading plus keeping the pipeline alive
after generation is a better tradeoff for this app.

### Per-model UI state matters

Two-step and four-step outputs are useful to compare. The app caches the last
image per model choice so switching models does not erase previous results.

## Current Runtime Defaults

```text
resolution = 512 x 512
guidanceScale = 1.0
schedulerTimestepShift = 3.0
seed = random
compute profile = aneFirst
prewarm = false
```

## Current Resource Strategy

```text
coremlsd3_2step/
coremlsd3_4step/
```

Each folder is generated locally and ignored by Git. The open-source repository
contains commands and scripts, not model weights.

## Recommended Reporting Format

When sharing a benchmark or bug report, include:

```text
iPhone model:
iOS version:
Xcode version:
model choice: 2-step or 4-step
stage sizes:
resource folder size:
first pipeline build time:
generation time:
last [MEM] log:
Core ML / ANE error:
```

This makes reports comparable across devices and conversion layouts.
