# ReSeed on the Apple Neural Engine

[ReSeed](https://twwinde.github.io/reseed-project-page/) restores seed diversity
in DMD-distilled few-step models by adding a learned per-channel bias to every
transformer block output (image and text streams; the last, context-pre-only
block has an image bias only) during the first sampler interval only. For SD3
Medium that is 72,192 parameters
([code](https://github.com/TWWinde/ReSeed),
[weights](https://huggingface.co/Wenwu2000/ReSeed)).

## One set of weights

The app's 4-step model is the ReSeed paper's SD3 Medium 4-step DMD student
(`students/sd3m/seed_42/checkpoint-2000`) with the adapter
`adapters/sd3m/seed_42/checkpoint-1900`. ReSeed off (alpha 0) runs the plain
student; ReSeed on scales the biases by alpha (0.5 / 1.0 / 1.5). Switching
alpha never reloads the pipeline.

## How it maps to split Core ML stages

The resource folders use one MMDiT block per Core ML stage (`Stage0` embeddings,
`Stage1..Stage24` = blocks 0..23), so every block output is a stage output:

- blocks 0..22: at step 1 the runtime adds `alpha * bias` to
  `latent_image_embeddings_out` / `token_level_text_embeddings_out` between
  stages (`ReSeed.swift`, `MultiModalDiffusionTransformer.swift`);
- block 23: `Stage24` takes a `reseed_image_bias` input, added to the last
  block's image output before `norm_out` (zeros at steps 2..4 or when off);
- `ReSeedBiases.json` in the resource folder holds the biases
  (`tools/ane_probe/scripts/export_reseed_biases.py`).

A split with several blocks per stage hides the inner block outputs, so ReSeed
is disabled (with a log line) when stage count != block count + 1.

## Building the resources

```bash
STUDENT=checkpoints/reseed/students/sd3m/seed_42/checkpoint-2000/student
ADAPTER=checkpoints/reseed/adapters/sd3m/seed_42/checkpoint-1900
ONES=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
python tools/ane_probe/scripts/convert_sd3m_split_qchunk.py --ckpt-path $STUDENT \
  --latent 128 --stage-sizes $ONES --q-chunks 16 --int8 \
  --with-conditioning --final-bias-input -o build_1024 --compile-into coremlsd3_reseed_1024
# 512: --latent 64 --q-chunks 4 ... --compile-into coremlsd3_reseed
# then copy TextEncoder, TextEncoder2, VAEDecoder, vocab.json, merges.txt from
# coremlsd3_4step/ and export the biases:
python tools/ane_probe/scripts/export_reseed_biases.py $ADAPTER coremlsd3_reseed_1024/ReSeedBiases.json
```

## Validation

`tools/ane_probe/scripts/check_reseed_split.py` (1024, t = 1000, real CLIP
conditioning) compares the Core ML stage chain with the PyTorch student plus the
paper's forward hooks:

| | relative L2 |
| --- | ---: |
| ReSeed effect in the reference (on vs off) | 0.394 |
| Core ML vs reference, ReSeed off | 0.109 |
| Core ML vs reference, ReSeed on | 0.096 |
| Core ML ReSeed delta vs reference delta | 0.245 |

The Core ML error is the usual fp16 / INT8 level; a misplaced bias would give a
delta error near or above 1.

On Mac (512, same prompt, seeds 1..4) alpha 0 gives near-identical frontal
portraits; alpha 1.0 and 1.5 vary pose, appearance, background and framing.

## iPhone 15 Pro (iOS 18.5)

| Resolution | First run (compile) | Warm | Peak app memory |
| --- | ---: | ---: | ---: |
| 512 | 66 s | 9.4 s (text 0.7, MMDiT 8.2, VAE 0.4) | ~58 MB |
| 1024 | 124 s | ~31 s (as without ReSeed) | ~126 MB |

The 512 one-block-per-stage ReSeed path is as fast as the original 4-blocks-per-
stage 512 resources (~9.5 s), so the finer split and the between-stage bias add
no measurable cost.
