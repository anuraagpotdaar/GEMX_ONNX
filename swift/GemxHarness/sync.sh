#!/bin/sh
set -e
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SRC=${BAMM_IOS_ROOT:-"$HERE/../../../BAMM_iOS"}/ARSmpl
DST="$HERE/Sources/GemxHarness/ios"
test -d "$SRC" || { echo "Missing BAMM_iOS sources: $SRC" >&2; exit 1; }
mkdir -p "$DST"
for f in Bamm/BvhMotion.swift Bamm/JointFrame.swift Pose/SkeletonTopology.swift Bamm/BammClient.swift \
         Bamm/HostedGemxClient.swift Bamm/GemxSettings.swift Bamm/OnDeviceGemxClient.swift Bamm/GemxPipeline.swift \
         Bamm/GemxVideo.swift Bamm/GemxMath.swift Bamm/GemxVitPoseCoreML.swift; do
  test -f "$SRC/$f" || { echo "Missing source: $SRC/$f" >&2; exit 1; }
  cp "$SRC/$f" "$DST/$(basename "$f")"
done
ls "$DST"
