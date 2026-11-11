#!/usr/bin/env bash
# Install ANEProbe on the connected iPhone and run each bundled single-block model in its own
# process launch (clean memory numbers). The stock-SDPA 1024 model runs last because it may be
# killed by jetsam. Output: results/<model>.log
set -uo pipefail
cd "$(dirname "$0")"
DEVICE="${DEVICE:?set DEVICE to your iPhone identifier (xcrun devicectl list devices)}"
APP=build/Build/Products/Release-iphoneos/ANEProbe.app
BUNDLE="${BUNDLE:-com.example.ANEProbe}"
MODELS="${MODELS:-r512_sdpa r1024_qchunk16 r1024_qchunk8 r1024_splitkv10 r1024_sdpa}"
PREDICT="${PREDICT:-5}"
mkdir -p results

xcrun devicectl device install app --device "$DEVICE" "$APP" || exit 1

for m in $MODELS; do
  for pass in 1 2; do   # pass 1 = first on-device ANE compile, pass 2 = cached
    echo "=== $m pass $pass"
    xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console \
      "$BUNDLE" --model "$m" --predict "$PREDICT" 2>&1 | grep --line-buffered "\[PROBE\]\|rror\|terminated\|crash" \
      | tee -a "results/$m.log"
    echo "exit=${PIPESTATUS[0]}" | tee -a "results/$m.log"
    sleep 3
  done
done
