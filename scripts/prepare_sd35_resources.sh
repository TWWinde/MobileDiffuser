#!/usr/bin/env bash
# Build the app-side SD3.5 512 resource folder.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${SD35_CKPT:-$ROOT/checkpoints/diffusion_pytorch_model.safetensors}"
BUILD="$ROOT/sd35_build_split_512"
RES="$ROOT/coremlsd35"
STAGES="${SD35_STAGE_SIZES:-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}"

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

echo "[1/4] Convert SD3.5 tokenizer/text/VAE resources"
"$ROOT/.venv/bin/python" "$ROOT/scripts/convert_sd35_components_coreml.py" \
  --repo stabilityai/stable-diffusion-3.5-medium \
  --build-dir "$ROOT/sd35_build_components_512" \
  --compile-into "$RES" \
  --latent-h 64 \
  --latent-w 64

echo "[2/4] Convert SD3.5 split transformer"
"$ROOT/.venv/bin/python" "$ROOT/scripts/convert_sd35_diffusers_split_coreml.py" \
  --ckpt-path "$CKPT" \
  --latent-h 64 \
  --latent-w 64 \
  --batch-size 1 \
  --stage-sizes "$STAGES" \
  -o "$BUILD"

echo "[3/4] Int8-quantize and compile split transformer into coremlsd35"
"$ROOT/.venv/bin/python" "$ROOT/scripts/quantize_mmdit_for_ane.py" \
  --split-dir "$BUILD" \
  --split-out-dir "$BUILD/int8" \
  --compile-into "$RES"

echo "[4/4] Summary"
du -sh "$RES" || true
