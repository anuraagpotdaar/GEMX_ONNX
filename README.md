# GEM-X on Apple devices

This repository contains GEM-X export, quantization and validation tools. The
companion [BAMM_iOS app](https://github.com/wangziyannb/BAMM_iOS) runs the
video-to-motion pipeline on iPhone. The optimization here targets its ViTPose
front end; it does not change the BAMM text-to-motion model.

## Supported ViTPose models

| App choice | Core ML package | Precision |
| --- | --- | --- |
| W8 | `quant/vitpose_w8.mlpackage` | INT8 weights, FP16 activations |
| W8A8 | `quant/vitpose_w8a8_clipped.mlpackage` | Same INT8 weights; INT8 activation boundaries around linear/conv operations |

W8A8 always includes the 32 SwiGLU range clamps needed for the observed Neural
Engine multiply/quantize execution issue. Other regions, including attention
matmul and normalization, remain floating point. These exports use the same
DINOv3 ViTPose Huge checkpoint and 77 SOMA keypoints.

Large checkpoints, model packages and compiled models are distributed separately
and remain ignored by Git. `model_artifacts.json` identifies the validated
packages by size and SHA-256. These are Core ML PTQ artifacts, not new PyTorch
training checkpoints.

## Rebuild the corrected W8A8 export

Use macOS, Python 3.11, `coremltools==9.0` and NumPy 1.26. The exporter validates
that version because it uses the 9.0 calibration hook. The baseline export also
needs the dependencies and ViTPose checkpoint from
[NVlabs/GEM-X](https://github.com/NVlabs/GEM-X).

Keep `GEM-X`, `GEMX_ONNX` and `BAMM_iOS` as sibling directories, or set
`GEMX_ROOT` and `BAMM_IOS_ROOT` explicitly. Run the following from this repository.

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
python -m pip install 'coremltools==9.0' 'numpy==1.26.4'

# Original PyTorch checkpoint: ../GEM-X/inputs/checkpoints/vitpose/vitpose.pth
python tools/export_vitpose_coreml.py

# Prepare crops with exactly the app's detector and preprocessing.
./swift/GemxHarness/sync.sh
swift build --package-path swift/GemxHarness -c release
swift/GemxHarness/.build/release/GemxHarness prepare-pose \
  /path/to/calibration.mp4 ../BAMM_iOS/ARSmpl/Resources/Gemx \
  runs/pose_calibration 2 32

python tools/quantize_vitpose_w8a8.py \
  --crops runs/pose_calibration/crops.f32
```

The PTQ script unions calibration extrema across all crops and their horizontal
flips, instead of overwriting earlier samples. It resumes cached calibration
only when the source graph, weights, input crops and tool version match. It
audits 291 activation quantizers, inserts all 32 SwiGLU clamps, verifies that W8
weight bytes are unchanged, and saves **only the corrected W8A8 package**. An
existing output is preserved; use `--output` and `--work-dir` for a separate run.

## Install models in the app bundle

```sh
xcrun coremlcompiler compile quant/vitpose_w8.mlpackage \
  ../BAMM_iOS/ARSmpl/Resources/Gemx
xcrun coremlcompiler compile quant/vitpose_w8a8_clipped.mlpackage \
  ../BAMM_iOS/ARSmpl/Resources/Gemx
```

Build the app with both `.mlmodelc` directories alongside the existing GEM-X
denoisers and skeleton tables. In GEM-X / On Device settings, select W8 or W8A8.
The default stays W8. No server or BAMM model change is needed.

## Validation

```sh
python -m unittest discover -s tests -p test_vitpose_w8a8.py -v
```

The regression checks cover accumulation across calibration samples, stale-cache
rejection and saturation preservation on CPU. The app's `posecompare` launch
plan compares W8 with corrected W8A8 using one warmup and three measured runs per
clip, alternating model order and recording thermal state and exported outputs.
Use held-out clips and matched settings. Report warm latency separately from
first-use compilation; output differences against a reference model are not
ground-truth pose accuracy.
