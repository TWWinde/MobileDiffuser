# iPhone OOM and ANE Debugging Notes

This document summarizes the practical debugging path for running split SD3
Medium Core ML resources on iPhone.

## Current Assumptions

- Resource folders: `coremlsd3_2step/` and/or `coremlsd3_4step/`.
- Resolution: 512 x 512.
- MMDiT: split into multiple `.mlmodelc` stages.
- MMDiT compression: INT8 linear symmetric weight quantization.
- Preferred compute units: `.cpuAndNeuralEngine`.
- Prewarm: disabled.

## Why OOM Happens

Common causes:

1. A Core ML model is too large for ANE plan compilation.
2. The app loads too many submodels at once.
3. A model falls back to GPU/CPU and expands compressed weights.
4. The resource folder accidentally mixes stages from different conversions.
5. A stale app install or on-device ANE cache keeps an old compiled plan.

The split-stage layout is meant to reduce per-stage ANE compiler pressure.
It does not remove the need for careful resource management.

## Useful Logs

The app prints memory checkpoints through `MemoryProbe`:

```text
[MEM] at app launch
[MEM] before pipeline build
[MEM] after pipeline build
[MEM] before generateImages
[MEM] step 1/4
[MEM] after MMDiT loop / before VAE decode
[MEM] after generateImages
```

If the app is killed, compare the last printed stage with the Xcode device log.

## How To Collect Jetsam Logs

On iPhone:

1. Open Settings.
2. Go to Privacy & Security.
3. Open Analytics & Improvements.
4. Open Analytics Data.
5. Search for `MobileDiffuser` or `JetsamEvent`.
6. Share the `.ips` file.

In Xcode:

1. Connect the iPhone.
2. Open Window -> Devices and Simulators.
3. Select the device.
4. Open View Device Logs.
5. Search for `MobileDiffuser` or `Jetsam`.

Important fields:

```text
reason
phys_footprint
rpages
largestProcess
frontmost
high_water_mark
```

`phys_footprint` is the most useful number because it is close to what jetsam
uses for process memory pressure decisions.

## ANE Compile Failures

Common messages include:

```text
ANE model load has failed
Must re-compile the E5 bundle
MILCompilerForANE error
ANECCompile() FAILED
MaxLiveInLiveOutExceeded
CompilationFailure
```

Suggested fixes:

1. Delete the app from the iPhone and reinstall.
2. Reboot the iPhone to clear stale compiler state.
3. Recompile with the same or newer iOS deployment target.
4. Split MMDiT into smaller stages.
5. Avoid eager prewarm.
6. Confirm all stages come from the same conversion run.

## Resource Consistency Check

Run:

```bash
find coremlsd3_2step -maxdepth 1 -mindepth 1 -print | sort
find coremlsd3_4step -maxdepth 1 -mindepth 1 -print | sort
du -sh coremlsd3_2step coremlsd3_4step
```

Expected:

- both folders have text encoders, VAE decoder, tokenizer files,
- both folders have MMDiT conditioning and stage models,
- each folder is several GB,
- no `.bak`, partial compile temp, or mixed stage directories are present.

## Debugging Checklist

1. Confirm Settings downloaded the selected resource folder, or that the app
   bundle contains a manually bundled resource folder.
2. Confirm `SD3PipelineLoader` prints split MMDiT stage count.
3. Confirm logs show `aneFirst`.
4. Generate once after a fresh install.
5. If it fails during pipeline build, reduce stage size.
6. If it fails during VAE decode, inspect decoder resource and memory.
7. If it falls back to CPU or GPU, treat it as a failed ANE validation run.

## Practical Stage Strategy

For 512 x 512, a medium split such as 6 body stages plus final output stage has
worked better than one huge model. Very small stages can compile more reliably
but may add scheduling overhead. Very large stages can be faster but are more
likely to hit ANE compiler limits.

The right split is device- and iOS-version-dependent. When reporting results,
include:

- iPhone model,
- iOS version,
- Xcode version,
- stage sizes,
- resource folder size,
- last memory log,
- ANE error message if present.
