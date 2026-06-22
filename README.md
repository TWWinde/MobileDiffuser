<h1>
  <img src="MobileDiffuser/Assets.xcassets/AppIcon.appiconset/AppIcon.png" alt="MobileDiffuser app icon" width="64" align="left">
  MobileDiffuser
</h1>

<br>

MobileDiffuser is an experimental iOS app for running distilled Stable
Diffusion 3 Medium locally on iPhone. The app targets 512 x 512 generation,
uses split Core ML MMDiT stages, and prefers Apple Neural Engine execution.

The repository contains the Swift app, a patched local copy of Apple's
`ml-stable-diffusion` package, conversion scripts, and documentation. It does
not contain model weights or compiled Core ML model bundles.

Mobile diffusion deployment is an active research area, but many published
systems do not release runnable mobile code. This project is open-sourced to
make on-device SD3 deployment easier to inspect, reproduce, and extend for
future researchers and builders.

## Screenshots

### Generated Images

<p>
  <img src="docs/images/showcase-collage.png" alt="Actual MobileDiffuser SD3 Medium 4-step local generation samples">
</p>

### Demo Video

[![MobileDiffuser demo video](https://img.youtube.com/vi/m5wlOkeY5Bw/maxresdefault.jpg)](https://www.youtube.com/shorts/m5wlOkeY5Bw)

### In-App Model Download

The GitHub repository and the installed app do not directly include model
weights or compiled Core ML bundles. After launch, open Settings and download
the selected SD3 Medium resources into the app sandbox. The downloaded files
are stored locally and reused across launches.

<p>
  <img src="docs/images/settings-download-2step.jpg" alt="MobileDiffuser Settings downloading SD3 Medium Core ML resources" width="280">
</p>

The Settings panel shows whether each model is missing or ready, and displays
the current file path while downloading large Core ML resource folders.

## Prebuilt Models

Prebuilt Core ML resources and the source distilled checkpoints are hosted on
Hugging Face:

[Wenwu2000/MobileDiffuser-SD3-medium](https://huggingface.co/Wenwu2000/MobileDiffuser-SD3-medium)

The app downloads these resources from its Settings panel after launch using
Swift networking against the Hugging Face API. Users do not need Python, pip,
Git LFS, or the Hugging Face CLI. Model files are not downloaded during Xcode
build and are not committed to this repository.

After in-app download, each local resource folder contains:

```text
coremlsd3_2step/TextEncoder.mlmodelc
coremlsd3_2step/TextEncoder2.mlmodelc
coremlsd3_2step/VAEDecoder.mlmodelc
coremlsd3_2step/MultiModalDiffusionTransformerStage0.mlmodelc

coremlsd3_4step/TextEncoder.mlmodelc
coremlsd3_4step/TextEncoder2.mlmodelc
coremlsd3_4step/VAEDecoder.mlmodelc
coremlsd3_4step/MultiModalDiffusionTransformerStage0.mlmodelc
```

The same Hugging Face repository also contains the source checkpoints under
`checkpoints/` for users who want to reproduce or modify the Core ML conversion.

## Current Status

- Model family: Stable Diffusion 3 Medium distilled checkpoints.
- App choices: `2 steps` and `4 steps`.
- Output size: 512 x 512.
- Runtime path: CLIP-L + CLIP-G text encoders, split MMDiT, VAE decoder.
- Guidance: CFG disabled in practice, `guidanceScale = 1.0`.
- Scheduler shift: `shift = 3.0`.
- Compute units: ANE-first (`cpuAndNeuralEngine`) for app validation.
- Quantization: INT8 linear symmetric weight quantization for split MMDiT.
- Resource folders expected by the app:
  - `coremlsd3_2step/`
  - `coremlsd3_4step/`

The resource folders are intentionally ignored by Git because each one is
roughly 2.7 GB.

## Performance

Observed 512 x 512 generation times on iPhone 15 Pro with ANE-first execution:

| Mode | Steps | Example generation time | Runtime memory after generation |
| --- | ---: | ---: | ---: |
| SD3 Medium 2-step | 2 | ~5.6 s | ~86 MB |
| SD3 Medium 4-step | 4 | ~9.5 s | ~87 MB |

These numbers are example measurements from local device testing. First use can
take longer because the app may need to download model resources and Core ML may
compile execution plans. Subsequent generations reuse the loaded pipeline when
possible.

## Repository Layout

```text
MobileDiffuser/
  ContentView.swift              SwiftUI UI and generation view model
  DiffusionModelKind.swift        2-step/4-step model selection
  SD3PipelineLoader.swift         Core ML pipeline loading and fallback logic
  MemoryProbe.swift               Lightweight runtime memory logging
  MobileDiffuser.entitlements     Increased memory limit entitlement

ml-stable-diffusion/
  Local patched Swift package used by the app.

scripts/
  convert_sd3_medium_split_coreml.py
  quantize_mmdit_for_ane.py
  test_sd3_two_step_mac.py
  and other conversion/debug helpers.

docs/
  ARCHITECTURE.md                 Runtime design and technical details
  REPRODUCING_MODELS.md           Step-by-step model conversion guide
  IPHONE_OOM_DEBUG.md             Historical iPhone memory notes
  TECHNICAL_REPORT.md             Longer experiment report
```

## Requirements

### For running the app

- macOS with Xcode 16.2 or newer.
- iOS 18.2 or newer deployment target.
- iPhone 15 Pro or newer is recommended.
- Apple Developer account for running on a physical iPhone.
- Network access on device for in-app model download, or manually bundled Core
  ML resources for offline development.

### For converting models

- Apple Silicon Mac.
- Python 3.11.
- At least 24 GB system memory recommended for conversion.
- Xcode command line tools.
- Access to the source checkpoint files.

## Quick Start

1. Clone the repository.

   ```bash
   git clone https://github.com/TWWinde/MobileDiffuser.git
   cd MobileDiffuser
   ```

2. Create the Python environment if you plan to convert models.

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ml-stable-diffusion
   pip install -r scripts/requirements.txt
   ```

3. Open `MobileDiffuser.xcodeproj` in Xcode.

4. Set your signing team.

   The open-source project intentionally uses:

   ```text
   PRODUCT_BUNDLE_IDENTIFIER = com.example.MobileDiffuser
   DEVELOPMENT_TEAM = ""
   ```

   In Xcode, select the `MobileDiffuser` target, choose your Team, and change
   the bundle identifier to something unique, for example:

   ```text
   com.yourname.MobileDiffuser
   ```

5. Build and run on a physical iPhone.

   The app is designed for device testing. Simulator is useful for UI only; it
   will not reproduce ANE behavior.

6. Download model resources in the app.

   Open the gear-shaped Settings panel and download either the selected model
   or both 2-step and 4-step resources. The app stores downloaded resources in
   its Application Support directory and reuses them across launches.

   Each downloaded folder contains:

   ```text
   TextEncoder.mlmodelc
   TextEncoder2.mlmodelc
   VAEDecoder.mlmodelc
   vocab.json
   merges.txt
   MultiModalDiffusionTransformerConditioning.mlmodelc
   MultiModalDiffusionTransformerStage0.mlmodelc
   MultiModalDiffusionTransformerStage1.mlmodelc
   ...
   MultiModalDiffusionTransformerStage6.mlmodelc
   ```

   See [docs/REPRODUCING_MODELS.md](docs/REPRODUCING_MODELS.md) for the full
   conversion flow.

## Model Conversion Summary

The fastest path for reproducing the current app resources is:

```bash
# 1. Convert the distilled SD3 Medium checkpoint into split fp16 mlpackages.
.venv/bin/python scripts/convert_sd3_medium_split_coreml.py \
  --ckpt-path checkpoints/diffusion_pytorch_model.safetensors \
  --latent-h 64 \
  --latent-w 64 \
  --batch-size 1 \
  --stage-sizes 4,4,4,4,4,4 \
  --ios-target iOS18 \
  -o sd3_four_step_build_split_512

# 2. INT8 quantize and compile the split MMDiT into the app resource folder.
.venv/bin/python scripts/quantize_mmdit_for_ane.py \
  --split-dir sd3_four_step_build_split_512 \
  --split-out-dir sd3_four_step_build_split_512/int8 \
  --compile-into coremlsd3_4step \
  --ios-deployment-target 18.2 \
  --mode linear_symmetric
```

You also need text encoder, VAE decoder, and tokenizer resources. These can be
converted with the upstream Core ML Stable Diffusion tooling or copied from a
compatible SD3 Medium resource folder:

```bash
cp -R coremlsd3_2step/TextEncoder.mlmodelc coremlsd3_4step/TextEncoder.mlmodelc
cp -R coremlsd3_2step/TextEncoder2.mlmodelc coremlsd3_4step/TextEncoder2.mlmodelc
cp -R coremlsd3_2step/VAEDecoder.mlmodelc coremlsd3_4step/VAEDecoder.mlmodelc
cp coremlsd3_2step/vocab.json coremlsd3_4step/vocab.json
cp coremlsd3_2step/merges.txt coremlsd3_4step/merges.txt
```

For a complete and more careful walkthrough, use
[docs/REPRODUCING_MODELS.md](docs/REPRODUCING_MODELS.md).

## Runtime Strategy

The app uses a memory-conscious pipeline:

1. Resolve the selected resource folder (`coremlsd3_2step` or
   `coremlsd3_4step`).
2. Load CLIP-L and CLIP-G text encoders.
3. Precompute timestep conditioning.
4. Execute split MMDiT stages sequentially.
5. Decode latents through the VAE decoder.
6. Keep the pipeline alive after generation so repeated generation avoids the
   full first-load cost.
7. Cache the last generated image per model choice, so switching from 2-step to
   4-step and back restores the previous image.

The split-stage design reduces per-model ANE compiler pressure. It does not
make the total model small; it makes each compiled sub-plan small enough to
load and execute more reliably on device.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## App Controls

- `2 steps`: uses `coremlsd3_2step` and `stepCount = 2`.
- `4 steps`: uses `coremlsd3_4step` and `stepCount = 4`.
- Prompt field: text prompt sent to CLIP encoders.
- Generate: runs the selected model.
- Share: exports the current generated image.

Generation uses random seeds by default. The selected seed is printed in the
debug log:

```text
[SD3] seed: 123456789
```

Set `config.seed` explicitly in `ContentView.swift` if you need deterministic
reproduction.

## Troubleshooting

### `resources not found`

Open the in-app Settings panel and download the selected model. The app looks
for downloaded resources in Application Support first, then falls back to
bundled resources if you added them manually for development.

### ANE compile or load failure

Common causes:

- The MMDiT stage is still too large.
- The model was compiled for an incompatible iOS/Core ML target.
- A stale on-device compiled ANE cache is being reused.
- The resource folder contains mixed files from different conversions.

Try:

- smaller stage sizes,
- `--ios-deployment-target 18.2`,
- deleting and reinstalling the app,
- rebooting the iPhone,
- regenerating the `.mlmodelc` folders cleanly.

### App is killed by memory pressure

Use Xcode device logs and the built-in memory log lines:

```text
[MEM] before pipeline build
[MEM] before generateImages
[MEM] step 1/4
[MEM] after generateImages
```

The app intentionally avoids eager prewarm because prewarming every Core ML
submodel can create a large initial memory spike before generation begins.

## Contributing

Contributions are welcome, especially:

- reproducible conversion notes for other SD3 distilled checkpoints,
- ANE compile/load failure reports with stage sizes and iOS version,
- memory measurements on different iPhone models,
- smaller or faster split-stage layouts,
- better UI and model management.

Please do not open pull requests that include model weights or compiled model
bundles. Share scripts, hashes, commands, and measurements instead.

## License

Code in this repository is intended to be released under the MIT License. Model
weights and converted Core ML assets are subject to their original model
licenses and are not included in this repository.

## Citation

If you use MobileDiffuser in your work, please cite:

```bibtex
@misc{tang2025mobilediffuser,
  author       = {Wenwu Tang and Dong Wang and Olga Saukh},
  title        = {MobileDiffuser: On-device Stable Diffusion 3 Medium on iPhone with Core ML},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/TWWinde/MobileDiffuser}},
  note         = {Accessed: 2026-06-19}
}
```
