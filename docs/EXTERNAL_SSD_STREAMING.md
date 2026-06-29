# External-SSD weight streaming on iPhone — feasibility & plan

**Status: GO, as a hardcore challenge experiment (not a shipping product).** Deferred — this doc
captures the research so we can pick it up later.

The idea: stream a model's transformer weights **block-by-block off an externally-powered USB-C SSD**
so an iPhone can run models far larger than its RAM *or* internal free space (FLUX.2 dev, Qwen-Image
20B), with **active cooling** removing thermal throttling. The point is to prove the
"impossible-seeming" thing works, not to ship it.

---

## 1. The decisive physics (why this is hard, and what it does/doesn't buy)

- **Block-streaming already removed the RAM ceiling.** Klein 4B 1024 streams at a ~3.8 GB peak from
  *internal* flash (validated on-device). So external SSD buys **STORAGE, never RAM**: its only
  benefit is fitting weights too big for internal free space.
- **The I/O tax is `transformer_size × steps ÷ throughput`** — every step re-preads the whole
  transformer (load→run→release per block). This penalizes exactly the big/high-step models that
  would justify external storage.
- Internal NAND ≈ 1 GB/s; iPhone **external** read is much lower (the iPhone USB stack, not the drive,
  is the bottleneck — see §3).

### Original verdict (product framing): **NO-GO**
As a shippable feature it's dead: `{too big for internal} ∩ {licensable} ∩ {tolerable latency} = ∅`
(iPhone 17 Pro/Max ship 1–2 TB internal; FLUX.2 dev is HF-gated/non-commercial; the one clean-license
big model, Qwen-Image Apache-2.0, is ~8–22 min/image). Plus power, thermal, reliability blockers.

### Reframed verdict (challenge experiment + hardware mitigations): **GO**
The user's setup removes the two **physical** blockers, and the experimental goal removes the
empty-product-use-case objection:
- **Externally-powered (TB4/USB-C) SSD** → kills the **4.5 W peripheral power cap** ("accessory uses
  too much power"). The drive doesn't draw from the phone.
- **AC-compressor active cooling on the phone** → kills **thermal throttling** of both the GPU clocks
  and the USB PHY/controller → sustained max throughput + compute for the whole multi-minute run.
- **Experiment, not product** → "slow but it works" (minutes/image) is the *win condition*, not a
  dealbreaker.

---

## 2. Code readiness ≈ 60% (what exists vs what's missing)

Already there (no change needed):
- **`RangedFileWeightSource`** (`swift-diffusion-core/Sources/DiffusionCore/Weights/RangedFileWeightSource.swift`)
  — POSIX **`pread`** of each tensor's exact byte range over an **arbitrary file URL**, `freesOnRelease
  == true`. Works on an external volume as-is. **pread (not mmap) is load-bearing**: on a mid-run
  unplug/unmount it throws a clean `shortRead`, not a crash. *This constraint must be preserved.*
- **`MemoryGovernor.plan(externalSSDAvailable:)`** → the `.streamingExternal` residency rung, plumbed
  end-to-end (`externalSSDAvailable` flows into `MLXDiffusionEngine.load()` ~line 88). Currently always
  `false` from the callers — cosmetic until the app flips it on.
- **`Flux2ComponentSource.open(modelDirectory:streaming:)`** and **`ZImageComponentSource.open(...)`**
  accept arbitrary URLs — no Application-Support assumption in the core libraries.

Missing (the ~450–550 lines of app-layer work):
- **Security-scoped access.** `RangedFileWeightSource` uses raw `open()` with **no
  `startAccessingSecurityScopedResource`**. iOS needs the security-scoped resource held (defer-balanced)
  for the *entire* run while preading the external folder.
- **App: pick + persist the location.** No `UIDocumentPicker` / bookmark code; `AppModel` hardwires the
  download base to internal Application Support (`AppModel.swift` ~lines 318–321). Need: pick an
  external folder → store a security-scoped **bookmark** → route the component source at it → set
  `externalSSDAvailable = true`.

---

## 3. The one real empirical unknown: iPhone USB-3 sustained read

- **iPhone has NO Thunderbolt.** iPhone 15/16 Pro USB-C = **USB 3.2 Gen 2 (10 Gbps)**. A TB4 drive runs
  at **USB 3** speeds on the phone — the **iPhone's USB stack is the bottleneck, not the drive**.
  - ⇒ The drive being TB4 doesn't help beyond the USB-3 ceiling; what matters is a **fast NVMe with
    USB-protocol fallback**. **A Thunderbolt-only enclosure won't be recognized by the iPhone** —
    confirm USB 3 fallback.
- Real sustained read on iPhone: ~**300 MB/s** measured with a slow SATA drive (Samsung T7); a fast
  NVMe enclosure is commonly ~**700–900 MB/s** on iPhone 15 Pro USB-3. With cooling preventing PHY
  throttle, it should hold near the ceiling. **This number is the project's pivotal unknown — measure
  it first (§5 spike).**

---

## 4. I/O numbers (thermal solved ⇒ sustained throughput)

`per-image pread = transformer_size × steps`. At a working estimate of **~800 MB/s**:

| Model | transformer / precision | steps | per-image I/O | @800 MB/s |
|---|---|---|---|---|
| Klein 4B 4-bit (baseline) | ~2.5 GB | 4 | ~10 GB | ~0.2 min — *but fits internal; no need to stream externally* |
| FLUX.2 dev fp8 | ~12 GB | 28 | ~336 GB | **~7 min** |
| FLUX.2 dev bf16 | ~24 GB | 28 | ~672 GB | **~14 min** |
| Qwen-Image 20B fp8 | ~14 GB | 28 | ~400 GB | **~8 min** |

Slow, but acceptable for a "prove it runs" experiment. Add compute on top (overlap via depth-1
prefetch could hide some I/O behind block compute — a worthwhile follow-up). At ~600 MB/s these grow
~1.3×; at ~300 MB/s (slow drive / USB 2 on non-Pro phones) ~2.7× → avoid slow drives.

---

## 5. Plan (spike first, then build)

**Step 0 — SPIKE the unknown (minimal code).** Throwaway test build: hold a security-scoped resource
over a folder on the powered external volume and run a **20-minute continuous `pread` loop** over a
large file. Measure **sustained MB/s** and confirm the **volume stays mounted under sustained load**.
- ≥ ~700 MB/s and stable → green, proceed.
- If it can't hold throughput or the volume drops → stop; the physics won't cooperate.

**Then build (~450–550 lines app-layer + a small core change):**
1. **Core:** wrap `RangedFileWeightSource`'s `open()` in `startAccessingSecurityScopedResource` /
   `stopAccessingSecurityScopedResource` (defer-balanced). **Keep `pread`, never mmap.**
2. **App:** `UIDocumentPicker` → external folder → persist security-scoped **bookmark**; resolve it at
   load, hold it for the run; point `Flux2ComponentSource.open(modelDirectory:)` /
   `ZImageComponentSource` at it; set `externalSSDAvailable = true` so `MemoryGovernor` picks
   `.streamingExternal`.
3. **(Optional) depth-1 prefetch** in the streaming loop (overlap block i+1's pread with block i's
   compute) to hide I/O behind compute — biggest lever for wall-clock.

**Model choice:**
- **Qwen-Image 20B (Apache-2.0)** — clean license, the best "runs *and* can be shown" target (~8 min/img).
- **FLUX.2 dev** — maximum challenge, but HF-gated + non-commercial (fine for a private experiment, not
  shippable).

---

## 6. Remaining risk (manageable in a controlled experiment)

- **Bookmark staleness / unmount mid-run.** fds are held for the whole 10–20 min run and the full
  transformer is preaded every step; a hot-unplug / iOS unmount / app suspension stales the bookmark →
  every later `pread` fails, killing the generation late. In a **controlled** setup (powered + plugged
  + app foreground + actively cooled) this essentially won't happen, and it fails as a clean thrown
  `shortRead`, not a crash — *as long as the loader stays on pread, not mmap.*

---

## 7. Hardware checklist
- iPhone 15 Pro / 16 Pro (USB-C **USB 3**; non-Pro iPhones are USB 2 → ~50 MB/s, don't bother).
- **USB-3-cable** (the in-box Apple cable is USB 2 — it would throttle to ~50 MB/s; use a USB 3 / TB
  cable).
- Externally-**powered** fast NVMe enclosure **with USB-protocol fallback** (not TB-only).
- Active cooling on the phone (the AC-compressor rig).
- Volume formatted APFS or exFAT, single data partition, model folder copied on.
