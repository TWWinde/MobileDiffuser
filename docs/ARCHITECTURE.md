# Architecture

This document explains how MobileDiffuser runs distilled SD3 Medium on iPhone
with Core ML.

## High-Level Pipeline

The app follows the SD3 inference structure:

```text
prompt
  -> CLIP-L text encoder
  -> CLIP-G text encoder
  -> token embeddings + pooled embeddings
  -> timestep conditioning
  -> split MMDiT stages
  -> noise prediction
  -> flow matching scheduler update
  -> VAE decoder
  -> UIImage
```

For distilled checkpoints the app uses:

```text
guidanceScale = 1.0
schedulerTimestepShift = 3.0
batchSize = 1
```

The current UI exposes two model choices:

```text
2 steps -> coremlsd3_2step, stepCount = 2
4 steps -> coremlsd3_4step, stepCount = 4
```

## Why Split MMDiT?

The SD3 Medium transformer is the largest part of the pipeline. Loading it as a
single Core ML program can exceed ANE compiler limits or create large memory
spikes during plan construction.

MobileDiffuser uses split MMDiT execution:

```text
MultiModalDiffusionTransformerConditioning.mlmodelc
MultiModalDiffusionTransformerStage0.mlmodelc
MultiModalDiffusionTransformerStage1.mlmodelc
...
MultiModalDiffusionTransformerStage6.mlmodelc
```

The conditioning model computes timestep/modulation inputs. The stage models
then execute consecutive groups of transformer blocks. Intermediate latent and
text embeddings are passed from one stage to the next.

This design reduces the size of each ANE compilation unit. It does not reduce
the mathematical cost of the model, but it makes on-device compilation and
loading more reliable.

## Why INT8 Weight Quantization?

The split MMDiT stages are converted from fp16 mlpackages and then compressed
with Core ML linear symmetric INT8 weight quantization.

The conversion script calls Core ML Tools compression in this style:

```bash
python scripts/quantize_mmdit_for_ane.py \
  --mode linear_symmetric
```

This is weight-only quantization. Activations are still handled by the Core ML
execution plan. The main benefits are:

- smaller stage bundles,
- less weight memory pressure,
- lower ANE load pressure,
- faster model transfer and plan construction in many cases.

Text encoders and VAE decoder are not necessarily quantized by this script.
The current script focuses on MMDiT because it dominates memory and compile
pressure.

## Scheduler

The app uses a distilled flow-matching schedule with shift applied once:

```text
t     = linspace(1.0, 0.0, stepCount + 1)
sigma = shift * t / (1 + (shift - 1) * t)
```

With `stepCount = 4` and `shift = 3.0`:

```text
t     = [1.0, 0.75, 0.5, 0.25, 0.0]
sigma = [1.0, 0.9, 0.75, 0.5, 0.0]
```

The scheduler update is:

```text
x_next = x + (sigma_next - sigma_current) * v
```

where `v` is the velocity/noise prediction from MMDiT.

## Model Loading Strategy

`SD3PipelineLoader` resolves one resource folder based on the selected UI
model. It checks for required resources before trying to build the pipeline.

The app currently validates the ANE path by using:

```swift
private let computeProfiles: [SD3PipelineLoader.ComputeUnitsProfile] = [.aneFirst]
```

`aneFirst` maps to:

```swift
.cpuAndNeuralEngine
```

There are additional profiles in the loader (`hybrid`, `gpuFirst`, `cpuOnly`)
for debugging and future fallback work, but the app intentionally keeps the
normal path ANE-first so slow CPU fallback does not hide an ANE failure.

## No Eager Prewarm

The app avoids eager prewarm. Prewarming all models at startup can compile too
many Core ML plans at once and create memory spikes before generation begins.

Instead:

1. The app validates resources at launch.
2. The first Generate builds the pipeline.
3. The pipeline stays alive for later generations.
4. The pipeline is unloaded on model switch, app backgrounding, memory warning,
   or generation failure.

## Per-Model Image Cache

The UI keeps one generated image snapshot per model choice:

```text
2-step image cache
4-step image cache
```

When the user switches from 2-step to 4-step and back, the previous 2-step
image is restored without re-running inference. This keeps model comparison
usable while still unloading the previous pipeline to avoid memory pressure.

## Memory Logging

The app prints memory checkpoints such as:

```text
[MEM] at app launch
[MEM] before pipeline build
[MEM] after pipeline build
[MEM] before generateImages
[MEM] step 1/4
[MEM] after generateImages
```

These logs are intentionally low-tech because they work on physical devices
through Xcode's console and make it clear whether a failure happens during:

- pipeline construction,
- text encoding,
- MMDiT stage execution,
- VAE decode,
- post-generation cleanup.

## Resource Folder Contract

Each resource folder must contain:

```text
TextEncoder.mlmodelc
TextEncoder2.mlmodelc
VAEDecoder.mlmodelc
vocab.json
merges.txt
MultiModalDiffusionTransformerConditioning.mlmodelc
MultiModalDiffusionTransformerStage0.mlmodelc
...
MultiModalDiffusionTransformerStageN.mlmodelc
```

The current app expects split MMDiT resources. A single unsplit
`MultiModalDiffusionTransformer.mlmodelc` is not accepted by the ANE-first
path because it is likely to fail or fall back to an undesired slow path.

## Known Limitations

- The GitHub repository does not include converted Core ML model folders; use
  the Hugging Face model repository or regenerate them locally.
- The app currently exposes 512 x 512 SD3 Medium 2-step and 4-step modes.
- Performance depends heavily on iOS version, device thermals, and Core ML ANE
  compiler behavior.
