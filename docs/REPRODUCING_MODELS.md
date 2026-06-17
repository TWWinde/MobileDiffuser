# Reproducing Model Resources

This guide explains how to regenerate the local Core ML resources used by
MobileDiffuser. Model weights are not included in the GitHub repository.

## Expected Output

The iOS app looks for resource folders at the repository root:

```text
coremlsd3_2step/
coremlsd3_4step/
```

Each folder is copied into the app bundle as a folder reference. The folder
name must be preserved.

## 0. Prepare Environment

Use Apple Silicon macOS and Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ml-stable-diffusion
pip install -r scripts/requirements.txt
```

Confirm Core ML compiler is available:

```bash
xcrun coremlcompiler --help
```

## 1. Prepare Checkpoints

Create a local checkpoint directory:

```bash
mkdir -p checkpoints
```

Place your distilled SD3 Medium transformer checkpoint there, for example:

```text
checkpoints/diffusion_pytorch_model.safetensors
```

The scripts also accept absolute paths through `--ckpt-path` or `--ckpt`.

Important: checkpoints are ignored by Git. Do not commit them.

## 2. Convert Split MMDiT

For a 512 x 512 SD3 Medium resource folder, use latent size 64 x 64.

The current app has used a 7-piece split:

```text
conditioning + Stage0...Stage6
```

Convert fp16 split mlpackages:

```bash
.venv/bin/python scripts/convert_sd35_diffusers_split_coreml.py \
  --ckpt-path checkpoints/diffusion_pytorch_model.safetensors \
  --model-family sd3-medium \
  --latent-h 64 \
  --latent-w 64 \
  --batch-size 1 \
  --stage-sizes 4,4,4,4,4,4 \
  --ios-target iOS18 \
  -o sd3_four_step_build_split_512
```

Notes:

- `--model-family sd3-medium` selects the SD3 Medium transformer structure.
- `--batch-size 1` matches distilled no-CFG inference.
- `--stage-sizes 4,4,4,4,4,4` groups the 24 transformer blocks into 6 body
  stages. The final projection appears as an additional output stage in the
  generated resources.
- If ANE compilation fails on device, try smaller stage sizes.

## 3. INT8 Quantize and Compile

Quantize the split MMDiT weights and compile the stages:

```bash
.venv/bin/python scripts/quantize_mmdit_for_ane.py \
  --split-dir sd3_four_step_build_split_512 \
  --split-out-dir sd3_four_step_build_split_512/int8 \
  --compile-into coremlsd3_4step \
  --ios-deployment-target 18.2 \
  --mode linear_symmetric
```

Expected result:

```text
coremlsd3_4step/MultiModalDiffusionTransformerConditioning.mlmodelc
coremlsd3_4step/MultiModalDiffusionTransformerStage0.mlmodelc
...
coremlsd3_4step/MultiModalDiffusionTransformerStage6.mlmodelc
```

## 4. Add Shared Text/VAE Resources

The app also needs:

```text
TextEncoder.mlmodelc
TextEncoder2.mlmodelc
VAEDecoder.mlmodelc
vocab.json
merges.txt
```

If you already have a compatible SD3 Medium resource folder, copy them:

```bash
cp -R coremlsd3_2step/TextEncoder.mlmodelc coremlsd3_4step/TextEncoder.mlmodelc
cp -R coremlsd3_2step/TextEncoder2.mlmodelc coremlsd3_4step/TextEncoder2.mlmodelc
cp -R coremlsd3_2step/VAEDecoder.mlmodelc coremlsd3_4step/VAEDecoder.mlmodelc
cp coremlsd3_2step/vocab.json coremlsd3_4step/vocab.json
cp coremlsd3_2step/merges.txt coremlsd3_4step/merges.txt
```

If you need to regenerate those resources from Hugging Face, use the upstream
Core ML Stable Diffusion converter through the patched local package:

```bash
.venv/bin/python -m python_coreml_stable_diffusion.torch2coreml \
  --model-version stabilityai/stable-diffusion-3-medium \
  --sd3-version \
  --convert-text-encoder \
  --convert-vae-decoder \
  --latent-h 64 \
  --latent-w 64 \
  --min-deployment-target iOS18 \
  -o sd3_build_components
```

Then compile/copy the produced text encoder and VAE decoder resources into the
target folder. Exact output names can vary with converter versions; inspect the
generated `sd3_build_components` directory.

## 5. Build a 2-Step Resource Folder

The app treats 2-step and 4-step as separate resource folders because they may
come from different distilled checkpoints.

If your two-step checkpoint has the same SD3 Medium architecture, run the same
split conversion but compile into `coremlsd3_2step`:

```bash
.venv/bin/python scripts/convert_sd35_diffusers_split_coreml.py \
  --ckpt-path checkpoints/diffusion_pytorch_model_2step.safetensors \
  --model-family sd3-medium \
  --latent-h 64 \
  --latent-w 64 \
  --batch-size 1 \
  --stage-sizes 4,4,4,4,4,4 \
  --ios-target iOS18 \
  -o sd3_two_step_build_split_512

.venv/bin/python scripts/quantize_mmdit_for_ane.py \
  --split-dir sd3_two_step_build_split_512 \
  --split-out-dir sd3_two_step_build_split_512/int8 \
  --compile-into coremlsd3_2step \
  --ios-deployment-target 18.2 \
  --mode linear_symmetric
```

Then add the shared text/VAE/tokenizer resources to `coremlsd3_2step`.

## 6. Verify Resource Folders

Check top-level contents:

```bash
find coremlsd3_2step -maxdepth 1 -mindepth 1 -print | sort
find coremlsd3_4step -maxdepth 1 -mindepth 1 -print | sort
du -sh coremlsd3_2step coremlsd3_4step
```

Each folder is expected to be several GB. Current local builds are around 2.7
GB per folder.

## 7. Add Folders to Xcode

In Xcode:

1. Right-click the project navigator.
2. Choose "Add Files to MobileDiffuser...".
3. Select `coremlsd3_2step` and/or `coremlsd3_4step`.
4. Use folder references so the blue folder name is preserved.
5. Enable target membership for `MobileDiffuser`.

The built app must contain:

```text
MobileDiffuser.app/coremlsd3_2step/
MobileDiffuser.app/coremlsd3_4step/
```

## 8. Build

For local source validation without signing:

```bash
xcodebuild \
  -project MobileDiffuser.xcodeproj \
  -scheme MobileDiffuser \
  -configuration Debug \
  -destination generic/platform=iOS \
  CODE_SIGNING_ALLOWED=NO \
  build
```

For device deployment, open Xcode and set your own signing team.

## 9. Device Validation

Run on a physical iPhone and watch Xcode logs:

```text
[SD3] requested fallback order: aneFirst
[SD3] using split MMDiT: 7 stages + fused AdaLN
[MEM] before pipeline build
[MEM] after pipeline build
[MEM] step 1/4
[MEM] after generateImages
```

If you see ANE compiler failures, regenerate with smaller stages or clear the
on-device app install and ANE cache by rebooting the device.

## Cleanup

After validating the app, large intermediate folders can be deleted:

```bash
rm -rf sd3_four_step_build_split_512
rm -rf sd3_two_step_build_split_512
```

Do not delete `coremlsd3_2step/` or `coremlsd3_4step/` if you still want to run
the app locally.
