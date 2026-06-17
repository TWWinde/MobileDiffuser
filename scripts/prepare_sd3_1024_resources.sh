#!/usr/bin/env bash
# Build the app-side SD3 1024 resource folder.
#
# 1024 cannot reuse the 512 MMDiT or VAE decoder: both have fixed latent/image
# shapes. This script prepares coremlsd3_1024 with shared tokenizer/text
# encoders plus a 128x128 latent split MMDiT. Provide a separately converted
# 1024 VAE decoder mlpackage through --vae-decoder when available.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${SD3_CKPT:-$ROOT/checkpoints/sd3_medium_distilled.safetensors}"
STAGES="${SD3_1024_STAGES:-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}"
SPLITKV_CHUNKS="${SD3_1024_SPLITKV_CHUNKS:-10}"
SDPA_QUERY_CHUNKS="${SD3_1024_SDPA_QUERY_CHUNKS:-16}"
BUILD="$ROOT/sd3_build_split_1024_fused6"
RES="$ROOT/coremlsd3_1024"
VAE_DECODER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --stage-sizes)
      STAGES="$2"
      shift 2
      ;;
    --vae-decoder)
      VAE_DECODER="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 2
fi

mkdir -p "$RES"

echo "[1/5] Copy shared SD3 text/tokenizer resources"
for name in TextEncoder.mlmodelc TextEncoder2.mlmodelc vocab.json merges.txt; do
  if [[ -e "$ROOT/coremlsd3/$name" ]]; then
    rm -rf "$RES/$name"
    cp -R "$ROOT/coremlsd3/$name" "$RES/$name"
  else
    echo "Missing shared resource: coremlsd3/$name" >&2
    exit 2
  fi
done

echo "[2/5] Convert 1024 split MMDiT (latent 128x128)"
"$ROOT/.venv/bin/python" "$ROOT/scripts/convert_mmdit_split_low_mem.py" \
  --ckpt-path "$CKPT" \
  --latent-h 128 \
  --latent-w 128 \
  --batch-size 1 \
  --stage-sizes "$STAGES" \
  --split-input-embedding \
  --sdpa-mode splitkv \
  --splitkv-chunks "$SPLITKV_CHUNKS" \
  --block-micro-stages \
  --sdpa-query-chunks "$SDPA_QUERY_CHUNKS" \
  -o "$BUILD"

echo "[3/5] Quantize and compile split MMDiT into coremlsd3_1024"
"$ROOT/.venv/bin/python" "$ROOT/scripts/quantize_mmdit_for_ane.py" \
  --split-dir "$BUILD" \
  --split-out-dir "$BUILD/int8" \
  --compile-into "$RES"

echo "[4/5] Install 1024 VAE decoder"
if [[ -n "$VAE_DECODER" ]]; then
  tmp="$RES/.vae_compile_tmp"
  rm -rf "$tmp" "$RES/VAEDecoder.mlmodelc"
  mkdir -p "$tmp"
  xcrun coremlcompiler compile "$VAE_DECODER" "$tmp" >/dev/null
  mv "$tmp"/*.mlmodelc "$RES/VAEDecoder.mlmodelc"
  rm -rf "$tmp"
else
  echo "Skipped. Pass --vae-decoder <1024_vae_decoder.mlpackage>."
  echo "Do not copy the 512 VAEDecoder.mlmodelc here; its shape is 64x64 -> 512."
fi

echo "[5/5] Summary"
du -sh "$RES" || true
