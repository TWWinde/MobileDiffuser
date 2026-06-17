#!/usr/bin/env python3
"""
6-bit palettization for SD3 CoreML models.

coremltools can only palettize .mlpackage (uncompiled) models, NOT .mlmodelc.
Workflow:
  1. Convert from PyTorch → .mlpackage  (torch2coreml)
  2. Palettize .mlpackage              (this script)
  3. Compile → .mlmodelc               (this script --compile)

Usage:
    # If you have .mlpackage files:
    python scripts/palettize_sd3.py --mlpackage-dir ~/sd3_mlpackages --compile

    # Preview:
    python scripts/palettize_sd3.py --mlpackage-dir ~/sd3_mlpackages --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "coremlsd3"

# Match longer names first (text_encoder_2 before text_encoder).
SUBMODULES = [
    ("text_encoder_2", "TextEncoder2.mlmodelc"),
    ("text_encoder", "TextEncoder.mlmodelc"),
    ("vae_decoder", "VAEDecoder.mlmodelc"),
    ("mmdit", "MultiModalDiffusionTransformer.mlmodelc"),
]


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1_073_741_824:.2f} GB"


def is_mlmodelc(path: Path) -> bool:
    return path.is_dir() and path.suffix == ".mlmodelc"


def is_mlpackage(path: Path) -> bool:
    return path.is_dir() and path.suffix == ".mlpackage"


def find_mlpackage(directory: Path, submodule: str) -> Path | None:
    matches = sorted(directory.glob(f"*_{submodule}.mlpackage"))
    return matches[0] if matches else None


def discover_mlpackages(directory: Path) -> list[tuple[str, Path, str]]:
    found: list[tuple[str, Path, str]] = []
    for submodule, output_name in SUBMODULES:
        package = find_mlpackage(directory, submodule)
        if package:
            found.append((submodule, package, output_name))
    return found


def check_mlmodelc_only(models_dir: Path) -> bool:
    mlmodelc = list(models_dir.glob("*.mlmodelc"))
    mlpackage = list(models_dir.glob("*.mlpackage"))
    return bool(mlmodelc) and not mlpackage


def print_mlmodelc_help() -> None:
    print(
        """
Error: coremlsd3 里只有已编译的 .mlmodelc，无法直接压缩。

coremltools 的 palettize 只支持 .mlpackage（未编译格式）。
请按以下步骤重新转换：

  # 1. 克隆转换工具
  git clone https://github.com/apple/ml-stable-diffusion.git
  cd ml-stable-diffusion
  pip install -r requirements.txt

  # 2. 转换出 .mlpackage（512x512，TextEncoder 自动 6-bit）
  python -m python_coreml_stable_diffusion.torch2coreml \\
    --model-version stabilityai/stable-diffusion-3-medium \\
    --sd3-version \\
    --convert-text-encoder --convert-vae-decoder --convert-mmdit \\
    --quantize-nbits 6 \\
    --latent-h 64 --latent-w 64 \\
    -o ~/sd3_mlpackages

  # 3. 用本脚本压缩 MMDiT 并编译到 coremlsd3
  python scripts/palettize_sd3.py \\
    --mlpackage-dir ~/sd3_mlpackages \\
    --output-dir coremlsd3 \\
    --compile
""",
        file=sys.stderr,
    )


def palettize_mlpackage(
    package_path: Path,
    nbits: int,
    dry_run: bool,
    granularity: str,
    group_size: int,
) -> Path:
    size_before = dir_size(package_path)
    print(f"  input:  {package_path.name}")
    print(f"  before: {format_gb(size_before)}")

    if dry_run:
        print("  dry-run: skipped")
        return package_path

    import coremltools as ct

    start = time.time()
    model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    op_kwargs: dict = {"mode": "kmeans", "nbits": nbits, "granularity": granularity}
    if granularity == "per_grouped_channel":
        op_kwargs["group_size"] = group_size
    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpPalettizerConfig(**op_kwargs),
        op_type_configs={"gather": None},
    )
    compressed = ct.optimize.coreml.palettize_weights(model, config=config)

    suffix = "palettized_grouped" if granularity == "per_grouped_channel" else "palettized"
    out_path = package_path.parent / f"{package_path.stem}_{suffix}.mlpackage"
    if out_path.exists():
        shutil.rmtree(out_path)
    compressed.save(str(out_path))

    elapsed = time.time() - start
    size_after = dir_size(out_path)
    saved_pct = (1 - size_after / size_before) * 100 if size_before else 0
    print(f"  output: {out_path.name}")
    print(f"  after:  {format_gb(size_after)} ({saved_pct:.0f}% smaller)")
    print(f"  time:   {elapsed:.1f}s")
    print(f"  spec:   {compressed.get_spec().specificationVersion} (8=iOS17, 9=iOS18)")
    return out_path


def compile_to_mlmodelc(
    package_path: Path,
    output_dir: Path,
    output_name: str,
    dry_run: bool,
) -> None:
    dest = output_dir / output_name
    print(f"  compile → {dest.name}")

    if dry_run:
        print("  dry-run: skipped")
        return

    import coremltools as ct

    if dest.exists():
        shutil.rmtree(dest)

    ct.models.utils.compile_model(str(package_path), destination_path=str(dest))
    print(f"  compiled: {format_gb(dir_size(dest))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Palettize SD3 .mlpackage models and optionally compile to .mlmodelc."
    )
    parser.add_argument(
        "--mlpackage-dir",
        type=Path,
        help="Directory containing *_mmdit.mlpackage etc. from torch2coreml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for compiled .mlmodelc (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Legacy alias; checked for .mlmodelc-only to show help message",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=6,
        choices=(2, 4, 6, 8),
        help="Palette bit width (default: 6)",
    )
    parser.add_argument(
        "--granularity",
        choices=("per_tensor", "per_grouped_channel"),
        default="per_grouped_channel",
        help=(
            "LUT granularity. 'per_grouped_channel' (default) requires iOS 18+ "
            "but lets the GPU/ANE consume LUT weights directly without "
            "dequantizing to fp16 at load time. 'per_tensor' is the iOS 17 "
            "format that always dequantizes on Apple Silicon."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=16,
        help="Channels per LUT when granularity=per_grouped_channel (default: 16)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[s[0] for s in SUBMODULES],
        metavar="SUBMODULE",
        help="Only process these submodules: mmdit, vae_decoder, text_encoder, text_encoder_2",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile palettized .mlpackage to .mlmodelc in --output-dir",
    )
    parser.add_argument(
        "--skip-palettize",
        action="store_true",
        help="Skip palettization, only compile existing .mlpackage (already quantized)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List models and sizes without processing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mlpackage_dir is None:
        if check_mlmodelc_only(args.models_dir.resolve()):
            print_mlmodelc_help()
            sys.exit(1)
        print(
            "Error: specify --mlpackage-dir with .mlpackage files from torch2coreml.\n"
            "Run with only .mlmodelc in coremlsd3 to see reconversion instructions.",
            file=sys.stderr,
        )
        sys.exit(1)

    mlpackage_dir = args.mlpackage_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not mlpackage_dir.is_dir():
        print(f"Error: not found: {mlpackage_dir}", file=sys.stderr)
        sys.exit(1)

    models = discover_mlpackages(mlpackage_dir)
    if args.only:
        models = [m for m in models if m[0] in args.only]

    if not models:
        print(f"Error: no *_{{mmdit,text_encoder,...}}.mlpackage in {mlpackage_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"MLPackage dir: {mlpackage_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Quantization:  {args.nbits}-bit kmeans, {args.granularity}", end="")
    if args.granularity == "per_grouped_channel":
        print(f" (group_size={args.group_size})")
    else:
        print()
    if args.compile:
        print("Compile:       yes → .mlmodelc")
    if args.dry_run:
        print("Mode:          dry-run")
    print()

    if args.compile and not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, (submodule, package_path, output_name) in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {submodule}")
        if args.skip_palettize:
            result_path = package_path
            print(f"  skip palettize, using {package_path.name}")
        else:
            result_path = palettize_mlpackage(
                package_path,
                args.nbits,
                args.dry_run,
                args.granularity,
                args.group_size,
            )

        if args.compile:
            compile_to_mlmodelc(result_path, output_dir, output_name, args.dry_run)
        print()

    if args.dry_run:
        print("Dry-run complete.")
    elif args.compile:
        print(f"Done. Compiled models are in {output_dir}")
        print("Next: Product → Clean Build Folder in Xcode, then run on device.")
    else:
        print("Palettization complete. Re-run with --compile to generate .mlmodelc for the app.")


if __name__ == "__main__":
    main()
