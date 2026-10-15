import json, sys
from pathlib import Path
B = Path("/Users/anuraag/Developer/GEMX_ONNX/runs/bench"); OUT = Path("/Users/anuraag/Developer/GEMX_ONNX/reports/T6_SAMPLE_TIMINGS.md")
order = ["test3_demo_6fps", "test3_demo_8fps", "test2_demo_6fps", "test2_demo_8fps", "test3", "test2", "test"]
runs = {p.stem: json.loads(p.read_text()) for p in B.glob("*.json")}
clips = [c for c in order if c in runs] + sorted(set(runs) - set(order))
def f(x, d=1): return "—" if x is None else f"{x:.{d}f}"
def mmss(x): return "—" if x is None else (f"{int(x//60)}m{int(round(x%60)):02d}s" if x >= 60 else f"{x:.1f}s")
L = []
L.append("# GEM-X sample_inputs timing — every clip × every version (M2 Pro 16 GB)\n")
L.append("## Clips\n\n| clip | res | fps | frames |\n|---|---|---|---|")
for c in clips:
    r = runs[c]; L.append(f"| {c} | {r['w']}×{r['h']} | {r['fps']} | {r['frames']} |")
L.append("\n## End-to-end wall time per version (video in → BVH out)\n")
vers = [("as_is_pytorch_postproc", "A. as-is (PyTorch denoiser + postproc, service default)"),
        ("onnx_fp32_coreml", "B. ONNX denoiser fp32 (CoreML EP, demo ONNX path)"),
        ("onnx_fp16_cpu", "C. ONNX denoiser fp16 (CPU EP)"),
        ("onnx_int8pc_cpu", "D. ONNX denoiser int8 per-channel (CPU EP)"),
        ("noimg_fp32_cpu", "E. no-imgfeat fp32 (no SAM3DB)"),
        ("noimg_fp16_cpu", "F. no-imgfeat fp16 (no SAM3DB)"),
        ("noimg_int8_cpu", "G. no-imgfeat int8 per-channel (no SAM3DB)"),
        ("edge_vitpose_int8_noimg_fp16", "H. on-device tier: VitPose int8 (CPU) + no-imgfeat fp16")]
L.append("| version | " + " | ".join(clips) + " |"); L.append("|---|" + "---|" * len(clips))
for k, name in vers:
    L.append(f"| {name} | " + " | ".join(mmss(runs[c]["totals_s"].get(k)) for c in clips) + " |")
L.append("\n## Shared stage times (s)\n")
stages = [("read_video_s", "decode video (per read)"), ("Detection (YOLOX+ByteTrack)", "YOLOX-X + ByteTrack (CoreML EP)"), ("VitPose 2D keypoints", "VitPose ViT-H (PyTorch MPS)"),
          ("sam3db_wall_s", "SAM-3D-Body ViT-H (PyTorch MPS, incl. load)"), ("preprocess_noimg_wall_s", "preprocess wall w/o SAM3DB"), ("bvh_export_s", "SOMA rest skeleton + BVH export")]
L.append("| stage | " + " | ".join(clips) + " |"); L.append("|---|" + "---|" * len(clips))
for k, name in stages:
    L.append(f"| {name} | " + " | ".join(f(runs[c]["stages"].get(k)) for c in clips) + " |")
L.append("| VitPose int8 ORT-CPU (s, ms/frame) | " + " | ".join(
    (f"{f(runs[c]['stages']['vitpose_int8'].get('t_run_s'))} ({f(runs[c]['stages']['vitpose_int8'].get('ms_per_frame'),0)})" if 't_run_s' in runs[c]['stages'].get('vitpose_int8', {}) else "ERR") for c in clips) + " |")
L.append("| peak RSS (GB) | " + " | ".join(f(runs[c]["peak_rss_mb"] / 1000, 1) for c in clips) + " |")
L.append("\n## Denoiser only: median run time (s) and PA-MPJPE vs PyTorch regression (mm)\n")
dv = ["pytorch_regression", "pytorch_postproc", "onnx_fp32_coreml", "onnx_fp32_cpu", "onnx_fp16_cpu", "onnx_fp16_coreml", "onnx_int8pc_cpu", "onnx_w8_cpu",
      "noimg_fp32_cpu", "noimg_fp32_coreml", "noimg_fp16_cpu", "noimg_int8_cpu", "edge_vitpose_int8_noimg_fp16_cpu"]
L.append("| denoiser version | " + " | ".join(clips) + " |"); L.append("|---|" + "---|" * len(clips))
for k in dv:
    cells = []
    for c in clips:
        v = runs[c]["versions"].get(k, {})
        if "error" in v: cells.append("ERR")
        elif "t_median_s" not in v: cells.append("—")
        else: cells.append(f"{v['t_median_s']:.2f}s (first {v['t_first_s']:.1f}s, load {v.get('t_load_s',0):.1f}s) / {f(v.get('pa_mm_vs_ref'))} mm")
    L.append(f"| {k} | " + " | ".join(cells) + " |")
L.append("\n## VitPose int8 vs MPS fp32 keypoints (px mean)\n")
L.append("| " + " | ".join(clips) + " |\n|" + "---|" * len(clips))
L.append("| " + " | ".join(f(runs[c]["stages"].get("vitpose_int8", {}).get("kp2d_px_mean_vs_mps"), 2) for c in clips) + " |")
errs = [(c, k, v["error"][:140]) for c in clips for k, v in runs[c]["versions"].items() if "error" in v]
if errs:
    L.append("\n## Errors\n"); L += [f"- {c} / {k}: `{e}`" for c, k, e in errs]
san = [(c, runs[c].get("bvh_sanity")) for c in clips if runs[c].get("bvh_sanity")]
L.append("\n## BVH sanity (as-is output)\n" + ("\n".join(f"- {c}: {s}" for c, s in san) if san else "- all clean"))
OUT.write_text("\n".join(L) + "\n"); print("\n".join(L))
