#!/usr/bin/env python3
"""
INT8 linear quantization for the MMDiT mlpackage — alternative to 6-bit
kmeans palettization that produces an mlmodelc the iPhone GPU can consume
WITHOUT fp16 dequantization at load time.

Why we need this:
    Our previous `palettize_sd3.py` used 6-bit kmeans palettization. On Mac
    M-series this is fine (compressed metal pipeline supports lookup-table
    weights). On iPhone GPU, however, palettize-lookup is dequantized to
    fp16 in VRAM during model load — a 1.4 GB mlmodelc balloons to ~3.7 GB
    of dirty memory and triggers jetsam OOM on iPhone 15 Pro.

INT8 linear quantization is consumed natively by both Apple's GPU shaders
and ANE — no expansion, ~1.8 GB final mlmodelc, ~30 min runtime on a
24 GB Mac.

Usage:
    python scripts/quantize_mmdit_for_ane.py \\
        --in  sd3_build_b1/Stable_Diffusion_..._mmdit.mlpackage \\
        --out sd3_build_b1/mmdit_int8.mlpackage \\
        --compile-into coremlsd3
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("quantize_mmdit_for_ane")


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def gb(num_bytes: int) -> str:
    return f"{num_bytes / 1_073_741_824:.2f} GB"


def numbered_suffix(path: Path, prefix: str) -> int:
    stem = path.stem
    if not stem.startswith(prefix):
        return -1
    return int(stem.removeprefix(prefix))


def cleanup_stale_compiled_split(destination_dir: Path, installed_stems: set[str]):
    patterns = [
        "MultiModalDiffusionTransformer.mlmodelc",
        "MultiModalDiffusionTransformerAdaLN*.mlmodelc",
        "MultiModalDiffusionTransformerStage*.mlmodelc",
    ]
    for pattern in patterns:
        for candidate in destination_dir.glob(pattern):
            if candidate.stem in installed_stems:
                continue
            logger.info(f"  removing stale compiled resource: {candidate.name}")
            shutil.rmtree(candidate)


def cleanup_stale_split_packages(output_dir: Path):
    for pattern in [
        "MultiModalDiffusionTransformerAdaLN*.mlpackage",
        "MultiModalDiffusionTransformerStage*.mlpackage",
    ]:
        for candidate in output_dir.glob(pattern):
            logger.info(f"  removing stale quantized package: {candidate.name}")
            shutil.rmtree(candidate)


def compile_mlpackage(
    input_pkg: Path,
    destination_dir: Path,
    deployment_target: str,
    target_stem: str | None = None,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = destination_dir / f".{input_pkg.stem}.compile-tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    subprocess.run(
        [
            "xcrun",
            "coremlcompiler",
            "compile",
            str(input_pkg),
            str(tmp_dir),
            "--platform",
            "iOS",
            "--deployment-target",
            deployment_target,
        ],
        check=True,
    )

    compiled = tmp_dir / f"{input_pkg.stem}.mlmodelc"
    if not compiled.exists():
        candidates = list(tmp_dir.glob("*.mlmodelc"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Cannot find compiled model in {tmp_dir}")
        compiled = candidates[0]

    target_name = target_stem or input_pkg.stem
    target = destination_dir / f"{target_name}.mlmodelc"
    if target.exists():
        shutil.rmtree(target)
        logger.info(f"  removed old mlmodelc: {target.name}")
    compiled.rename(target)
    shutil.rmtree(tmp_dir)
    return target


def quantize_one(input_pkg: Path, output_pkg: Path, mode: str):
    import coremltools as ct

    logger.info(f"Loading mlpackage: {input_pkg.name}")
    logger.info(f"  size before: {gb(dir_size(input_pkg))}")

    # CPU-only load is intentional: weight ops are inspected on CPU; this
    # avoids spinning up GPU/ANE compilation we don't need yet.
    model = ct.models.MLModel(str(input_pkg), compute_units=ct.ComputeUnit.CPU_ONLY)

    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode=mode,
            dtype="int8",
            granularity="per_channel",   # per-channel quant — ANE supports this
        ),
        # Embedding / lookup-style ops don't benefit from linear quant — keep fp16.
        op_type_configs={"gather": None},
    )

    logger.info(f"Running linear_quantize_weights (mode={mode}, dtype=int8)…")
    t0 = time.time()
    quantized = ct.optimize.coreml.linear_quantize_weights(model, config=config)
    logger.info(f"Quantized in {time.time() - t0:.1f}s")

    if output_pkg.exists():
        shutil.rmtree(output_pkg)
    output_pkg.parent.mkdir(parents=True, exist_ok=True)
    quantized.save(str(output_pkg))
    size_after = dir_size(output_pkg)
    logger.info(f"Saved: {output_pkg.name}")
    logger.info(f"  size after:  {gb(size_after)}")
    logger.info(f"  ratio:       {dir_size(input_pkg) / size_after:.2f}× smaller")
    return output_pkg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="input_pkg",
        type=Path,
        default=Path(
            "sd3_build_b1/"
            "Stable_Diffusion_version_stabilityai_stable-diffusion-3-medium_mmdit.mlpackage"
        ),
        help="Path to the fp16 MMDiT .mlpackage from convert_mmdit_low_mem.py",
    )
    parser.add_argument(
        "--out",
        dest="output_pkg",
        type=Path,
        default=Path("sd3_build_b1/mmdit_int8.mlpackage"),
        help="Output INT8-quantized .mlpackage path",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing split MMDiT mlpackages "
            "(MultiModalDiffusionTransformerConditioning/Stage*.mlpackage)."
        ),
    )
    parser.add_argument(
        "--split-out-dir",
        type=Path,
        default=Path("sd3_build_split/int8"),
        help="Output directory for quantized split mlpackages.",
    )
    parser.add_argument(
        "--compile-into",
        type=Path,
        default=None,
        help=(
            "If set, also compile the INT8 mlpackage into "
            "<dir>/MultiModalDiffusionTransformer.mlmodelc (atomic replace)."
        ),
    )
    parser.add_argument(
        "--ios-deployment-target",
        default="18.2",
        help="Deployment target passed to xcrun coremlcompiler when compiling.",
    )
    parser.add_argument(
        "--mode",
        choices=("linear", "linear_symmetric"),
        default="linear_symmetric",
        help="Quantization range fitting (symmetric is ANE-friendly).",
    )
    args = parser.parse_args()

    if args.split_dir is not None:
        if not args.split_dir.exists():
            logger.error(f"Split directory not found: {args.split_dir}")
            return 2
        packages = [
            args.split_dir / "MultiModalDiffusionTransformerConditioning.mlpackage",
        ]
        packages += sorted(
            args.split_dir.glob("MultiModalDiffusionTransformerStage*.mlpackage"),
            key=lambda path: numbered_suffix(path, "MultiModalDiffusionTransformerStage"),
        )
        packages = [pkg for pkg in packages if pkg.exists()]
        if len(packages) < 2:
            logger.error(f"No complete split MMDiT package set found in {args.split_dir}")
            return 2

        cleanup_stale_split_packages(args.split_out_dir)
        installed_stems: set[str] = set()
        for input_pkg in packages:
            output_pkg = args.split_out_dir / input_pkg.name
            quantized_pkg = quantize_one(input_pkg, output_pkg, args.mode)
            if args.compile_into:
                logger.info(f"Compiling {quantized_pkg.name} into {args.compile_into} …")
                target = compile_mlpackage(
                    quantized_pkg,
                    args.compile_into,
                    args.ios_deployment_target,
                    target_stem=input_pkg.stem,
                )
                installed_stems.add(target.stem)
                logger.info(f"  installed: {target.name} ({gb(dir_size(target))})")
        if args.compile_into:
            cleanup_stale_compiled_split(args.compile_into, installed_stems)
        logger.info("Done.")
        return 0

    if not args.input_pkg.exists():
        logger.error(f"Input mlpackage not found: {args.input_pkg}")
        return 2

    quantize_one(args.input_pkg, args.output_pkg, args.mode)

    if args.compile_into:
        logger.info(f"Compiling into {args.compile_into} …")
        target = compile_mlpackage(
            args.output_pkg,
            args.compile_into,
            args.ios_deployment_target,
            target_stem="MultiModalDiffusionTransformer",
        )
        logger.info(f"  installed: {gb(dir_size(target))} INT8 mlmodelc")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
