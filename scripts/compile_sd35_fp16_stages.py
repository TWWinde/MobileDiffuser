#!/usr/bin/env python3
"""Compile SD3.5 fp16 split MMDiT packages into the app resource folder."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quantize_mmdit_for_ane import compile_mlpackage, dir_size, gb, numbered_suffix


LOG = logging.getLogger("compile_sd35_fp16_stages")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=Path("sd35_build_split_512"))
    parser.add_argument("--compile-into", type=Path, default=Path("coremlsd35"))
    parser.add_argument("--ios-deployment-target", default="iOS18")
    args = parser.parse_args()

    packages = [args.split_dir / "MultiModalDiffusionTransformerConditioning.mlpackage"]
    packages += sorted(
        args.split_dir.glob("MultiModalDiffusionTransformerStage*.mlpackage"),
        key=lambda path: numbered_suffix(path, "MultiModalDiffusionTransformerStage"),
    )
    packages = [package for package in packages if package.exists()]

    if len(packages) != 26:
        LOG.error("Expected 26 fp16 packages, found %d in %s", len(packages), args.split_dir)
        for package in packages:
            LOG.error("  found: %s", package.name)
        return 2

    LOG.info("Compiling %d fp16 SD3.5 MMDiT resources into %s", len(packages), args.compile_into)
    for package in packages:
        LOG.info("Compiling %s", package.name)
        target = compile_mlpackage(
            package,
            args.compile_into,
            args.ios_deployment_target,
            target_stem=package.stem,
        )
        LOG.info("  installed %s (%s)", target.name, gb(dir_size(target)))

    LOG.info("Done. Resource folder size: %s", gb(dir_size(args.compile_into)))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
