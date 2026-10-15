#!/bin/sh
set -e
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
CLIP=${1:?clip}; MODE=${2:-onDevice}; URL=${3:-}; SECS=${4:-150}; VIEW=${5:-3d}
SCRATCH=${TMPDIR:-/tmp}/gemx_sim
DD=$SCRATCH/DerivedData
OUT=/Users/anuraag/Developer/GEMX_ONNX/runs/sim; mkdir -p "$OUT"
cd /Users/anuraag/Developer/BAMM_iOS
xcodebuild -project ARSmpl.xcodeproj -scheme ARSmpl -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath "$DD" CODE_SIGNING_ALLOWED=NO -quiet build 2>&1 | grep -E "error:" || true
APP=$DD/Build/Products/Debug-iphonesimulator/ARSmpl.app
UDID=$(xcrun simctl list devices available | grep "iPhone 18 Pro (" | head -1 | sed -E 's/.*\(([0-9A-F-]+)\).*/\1/')
echo "simulator $UDID"
xcrun simctl boot "$UDID" 2>/dev/null || true
xcrun simctl bootstatus "$UDID" -b >/dev/null
xcrun simctl terminate "$UDID" com.example.arsmpl 2>/dev/null || true
xcrun simctl install "$UDID" "$APP"
xcrun simctl launch "$UDID" com.example.arsmpl -gemxAutoVideo "$CLIP" -gemxAutoMode "$MODE" -gemxAutoUrl "$URL" -gemxAutoView "$VIEW" -gemxAutoCycle "${CYCLE:-0}" -gemxAutoRig "${RIG:-smpl}" -gemxAutoStep "${STEP:-3}" -gemxAutoFlip "${FLIP:-0}" >/dev/null
echo "launched $(date +%T), streaming log for ${SECS}s"
LOG=$OUT/sim_$(basename "$CLIP" .mp4)_${MODE}_$VIEW.log
xcrun simctl spawn "$UDID" log stream --style compact --predicate 'process == "ARSmpl"' > "$OUT/raw.log" 2>&1 &
LP=$!
sleep $((SECS / 2))
xcrun simctl io "$UDID" screenshot "$OUT/sim_$(basename "$CLIP" .mp4)_${MODE}_${VIEW}_mid.png" >/dev/null
sleep $((SECS - SECS / 2))
kill $LP 2>/dev/null || true
grep -E "GemxAuto|RigPoseApplier|SceneCoordinator\]|Overlay\]" "$OUT/raw.log" | sed -E 's/^[^ ]+ [^ ]+ //' > "$LOG"
xcrun simctl io "$UDID" screenshot "$OUT/sim_$(basename "$CLIP" .mp4)_${MODE}_$VIEW.png" >/dev/null
echo "log: $LOG"; cat "$LOG"
