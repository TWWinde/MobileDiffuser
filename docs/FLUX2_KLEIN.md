# FLUX.2 Klein 4B on the Apple Neural Engine

FLUX.2 Klein (4B, 4-step distilled) runs in the app with every component as static-shape Core ML
stages on the ANE. One stage is loaded, run and released at a time.

| Part | Conversion | Stages |
| --- | --- | --- |
| Text encoder (Qwen3-4B) | first 27 of 36 layers, 3 per stage, INT8, 256 tokens | `TextStage0..8` |
| Denoiser (dim 3072) | one block per stage, INT8, exact query-chunked attention | `Embed`, `Double0..4`, `Single0..19`, `Final` |
| VAE decoder | 64x64x32 latent tile -> 512x512, chunked mid-block attention | `KleinVAEDecoder` |

Swift pipeline: `ml-stable-diffusion/swift/StableDiffusion/pipeline/Flux2KleinPipeline.swift`.

## Measurements (iPhone 15 Pro, iOS 18.5, 1024 x 1024, 4 steps, in the app)

| Phase | First run | Warm |
| --- | ---: | ---: |
| Text encoding | 46.9 s | 2.3 s |
| Denoising | 208.9 s | 69.9 s |
| VAE (3x3 tiles) | 7.7 s | 4.7 s |
| Total | ~264 s | ~77 s |

The first run includes ~194 s of one-time ANE compilation (37 stages). App memory stays around
185 MB. Single-stage probes: text stage 23 ms, double / single block ~500 ms isolated (~640 / 620 ms
inside the chain), VAE tile 327 ms; all ops on the ANE.

512 x 512 stages are converted and verified on the Mac only (single 512 blocks measured 80 / 91 ms
on the iPhone); the full 512 chain has not been timed on device. The 2-step option runs the same
4-step weights on a 2-step schedule.

## Fast mode: low-resolution start, high-resolution finish

A full 1024 step costs ~17.5 s on the phone, a 512 step ~2.7 s. The app's default for Klein at 1024
("Fast (512 → 1024)") runs the first 3 steps of the 512 schedule on the 512 stages, upsamples the
clean-image estimate x2 (bicubic, in the unpatchified VAE latent space), re-noises it to the sigma where
the last 1024 step starts (0.767), and runs only that last step on the 1024 stages. No retraining, no
new conversion — both stage sets already exist.

| Mode (iPhone 15 Pro, warm, in the app) | Text | Denoise | VAE | Total |
| --- | ---: | ---: | ---: | ---: |
| Full 1024, 4 steps | 2.3 s | 69.9 s | 4.7 s | ~77 s |
| Fast (3 x 512 + 1 x 1024) | 1.7 s | 26.1-26.8 s | 4.7 s | ~33 s |

Checked by eye on three prompts against full 1024 (sharp, no repeated texture, neon lettering
correct); not a systematic evaluation. The composition comes from the 512 steps, so a seed does not
give the same image as full 1024. The first fast run also compiles the 512 stages (once).

Other measured engineering points:

- The next stage is loaded on a background queue while the current one runs (two stages resident);
  load wait per image drops to ~0.1 s.
- Stage outputs are already float16; passing them on costs nothing.
- Running a stage right after a *different* stage costs ~50-90 ms more than repeating the same one
  (533 ms vs 587-617 ms for a 1024 single block), even with both resident: an ANE model-switch cost
  that comes with one block per stage. Keeping six 1024 stages resident gets the app killed.

## What needed care

- **Text length.** Klein pads prompts to 512 tokens; 256 gives visually equivalent images
  (LPIPS 0.15 vs 512) and makes every block cheaper.
- **Qwen3 in fp16.** From layer 6 on, the first token carries a ~1.6e4 activation. Its square
  overflows half precision, so RMSNorm divides by the per-token max |x| first, and the MLP scales the
  product by 1/32 before `down_proj` and by 32 after it; without the latter the ANE (not the CPU)
  returns inf.
- **Hidden states.** The prompt embedding is the concatenation of the hidden states after layers
  9 / 18 / 27, so layers 28-36 are never converted.
- **Tokenizer.** swift-transformers 0.1.8 has no `Qwen2Tokenizer` entry (loaded as generic BPE), wants
  merges as `"a b"` strings, and mis-splits a special token that follows an emoji; the chat template
  is therefore encoded as prefix+prompt and suffix separately. Token ids match the Python tokenizer.
- **ANE compiler stalls.** A stalled `ANECompilerService` makes later loads hang until the iPhone is
  rebooted; the same stage compiled normally afterwards.

Numerics on the Mac ANE: Core ML denoiser vs PyTorch LPIPS 0.10-0.20 over three prompts (INT8 + fp16);
Core ML text embeddings change the image by LPIPS ~0.11; the Swift pipeline matches the Python chain
(LPIPS 0.0002).

## Converting

```bash
S=tools/ane_probe/scripts; M=checkpoints/flux2-klein-4b; O=coreml_flux2_klein_1024
python $S/convert_flux2_klein_split.py --model $M --res 1024 --txt 256 --q-chunks 16 --int8 -o build --compile-into $O
python $S/convert_qwen3_split.py       --model $M --txt 256 --int8 --embeddings -o build --compile-into $O
python $S/convert_flux2_vae.py         --model $M --compile-into $O
cp $M/tokenizer/tokenizer.json $M/tokenizer/tokenizer_config.json $O/
# 512: denoiser only (--res 512 --q-chunks 3 into coreml_flux2_klein_512); text encoder, VAE and
# tokenizer are looked up in the sibling coreml_flux2_klein_1024 folder.
```

Mac checks (one stage resident at a time, with a memory watchdog — loading all stages at once
together with PyTorch exhausts a 24 GB Mac):

```bash
python $S/run_flux2_klein_lowmem.py denoise --model $M --coreml $O --embeds embeds.pt --prompt 0 --out out
python $S/run_flux2_klein_lowmem.py decode  --model $M --prompt 0 --out out
swift run -c release --package-path tools/sd3_ane_cli sd3-ane-cli --klein 1 --resources $O --prompt "..." --out out.png
```

Prebuilt folders are on Hugging Face (`coreml_flux2_klein_1024/`, `coreml_flux2_klein_512/`).
FLUX.2 Klein 4B and Qwen3-4B are Apache-2.0.
