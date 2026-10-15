#!/bin/sh
set -e
SRC=/Users/anuraag/Developer/BAMM_iOS/ARSmpl
DST="$(dirname "$0")/Sources/GemxHarness/ios"
rm -rf "$DST"; mkdir -p "$DST"
for f in Bamm/BvhMotion.swift Bamm/JointFrame.swift Pose/SkeletonTopology.swift Bamm/BammClient.swift \
         Bamm/HostedGemxClient.swift Bamm/GemxSettings.swift Bamm/OnDeviceGemxClient.swift Bamm/GemxPipeline.swift \
         Bamm/GemxVideo.swift Bamm/GemxMath.swift Bamm/GemxVitPoseCoreML.swift; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/$(basename "$f")"
done
ls "$DST"
