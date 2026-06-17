#!/usr/bin/env bash
# Compile SD3 mlpackages to mlmodelc + download tokenizer assets.
# Replaces the role of `--bundle-resources-for-swift-cli` without
# loading the SDXL Python pipeline (saves ~10 GB RAM).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/sd3_build"
RES="$BUILD/Resources"
PREFIX="Stable_Diffusion_version_stabilityai_stable-diffusion-3-medium"

mkdir -p "$RES"

palettized_mmdit="$BUILD/${PREFIX}_mmdit_palettized.mlpackage"
fp16_mmdit="$BUILD/${PREFIX}_mmdit.mlpackage"
if [[ -d "$palettized_mmdit" ]]; then
  echo "[1/5] Replacing fp16 MMDiT with 6-bit palettized version"
  rm -rf "$fp16_mmdit"
  mv "$palettized_mmdit" "$fp16_mmdit"
fi

compile_one() {
  local src="$1" target_name="$2"
  if [[ ! -d "$src" ]]; then
    echo "  -- skip $target_name (source $src missing)"
    return
  fi
  echo "  -- $target_name <- $(basename "$src")"
  local tmp="$RES/__compile_tmp"
  rm -rf "$tmp"
  mkdir -p "$tmp"
  xcrun coremlcompiler compile "$src" "$tmp" >/dev/null
  rm -rf "$RES/${target_name}.mlmodelc"
  mv "$tmp"/*.mlmodelc "$RES/${target_name}.mlmodelc"
  rm -rf "$tmp"
}

echo "[2/5] Compiling all mlpackage -> mlmodelc"
compile_one "$BUILD/${PREFIX}_text_encoder.mlpackage"    "TextEncoder"
compile_one "$BUILD/${PREFIX}_text_encoder_2.mlpackage"  "TextEncoder2"
compile_one "$BUILD/${PREFIX}_vae_decoder.mlpackage"     "VAEDecoder"
compile_one "$BUILD/${PREFIX}_mmdit.mlpackage"           "MultiModalDiffusionTransformer"

echo "[3/5] Downloading tokenizer vocab.json + merges.txt"
curl -fsSL "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json"  -o "$RES/vocab.json"
curl -fsSL "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt"  -o "$RES/merges.txt"

echo "[4/5] Replacing project-level coremlsd3/"
rm -rf "$ROOT/coremlsd3"
cp -R "$RES" "$ROOT/coremlsd3"

echo "[5/5] Summary"
du -sh "$ROOT/coremlsd3"/*
echo
echo "Total app payload:"
du -sh "$ROOT/coremlsd3"
