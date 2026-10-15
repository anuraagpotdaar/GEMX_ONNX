import argparse, gc, sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import common as C

ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--skip-vitpose-int8", action="store_true")
a = ap.parse_args()
video = Path(a.video).resolve(); stem = video.stem
demo = C.load_demo(); demo._ensure_pipeline_deps()
import torch, hydra, cv2
from gem.utils.video_io_utils import get_video_lwh
L, W, H = get_video_lwh(str(video))
fps = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) + 0.5) or 30
out_root = C.RUNS / "bench" / stem
cache = out_root / stem / "preprocess"
R = {"video": str(video), "stem": stem, "frames": L, "w": W, "h": H, "fps": fps, "stages": {}, "versions": {}}
def log(*s): print("[t6]", *s, flush=True)

t0 = time.time(); _ = C.read_frames(video); R["stages"]["read_video_s"] = time.time() - t0; del _; gc.collect()
cfg = C.build_cfg(demo, video, out_root, static_cam=True)
demo._timings.clear()
t0 = time.time(); demo.run_preprocess_fast(cfg, no_imgfeat=True); R["stages"]["preprocess_noimg_wall_s"] = time.time() - t0
t0 = time.time(); demo.run_preprocess_fast(cfg, no_imgfeat=False); R["stages"]["sam3db_wall_s"] = time.time() - t0
R["stages"].update({k: float(v) for k, v in demo._timings.items()})
R["stages"]["rss_after_preprocess_mb"] = C.rss_mb()
log("stages", {k: round(v, 1) for k, v in R["stages"].items()})

dev = demo.DEVICE
data_full = C.data_dict_from_cache(demo, video, cache, no_imgfeat=False)
data_noimg = C.data_dict_from_cache(demo, video, cache, no_imgfeat=True)
model = hydra.utils.instantiate(cfg.model, _recursive_=False)
model.load_pretrained_model(cfg.ckpt_path or "inputs/pretrained/gem_soma.ckpt")
model = model.eval().to(dev); ende = model.endecoder
if ende.obs_indices_dict is None: ende.build_obs_indices_dict()
from gem.utils.cam_utils import compute_transl_full_cam
from gem.pipeline.gem_pipeline import get_body_params_w_Rt_v2
soma = C.soma_layer()
tonp = lambda d: {k: v.detach().float().cpu().numpy() for k, v in d.items() if isinstance(v, torch.Tensor)}

def decode_onnx(out, data):
    pred_x = torch.from_numpy(out["pred_x"]).to(dev); pred_cam = torch.from_numpy(out["pred_cam"]).to(dev)
    dd = ende.decode(pred_x)
    sp = dd["scale_params"].clone(); sp[..., 0] = sp[..., 0].clamp(0.7, 1.0); dd["scale_params"] = sp
    g = get_body_params_w_Rt_v2(global_orient_gv=dd["global_orient_gv"], local_transl_vel=dd["local_transl_vel"],
                                global_orient_c=dd["global_orient"], cam_angvel=data["cam_angvel"][None].to(dev))
    return {"body_pose": dd["body_pose"][0], "global_orient": g["global_orient"][0], "transl": g["transl"][0],
            "identity_coeffs": dd["identity_coeffs"][0], "scale_params": dd["scale_params"][0]}

def feed_of(data):
    f = {"obs": data["kp2d"][None].numpy(), "bbx_xys": data["bbx_xys"][None].numpy(), "K_fullimg": data["K_fullimg"][None].numpy(),
         "f_imgseq": data["f_imgseq"][None].numpy(), "f_cam_angvel": data["cam_angvel"][None].numpy()}
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in f.items()}

def pytorch_version(name, data, postproc):
    ts = []
    for _ in range(a.repeat):
        t0 = time.time()
        with torch.no_grad(): pred = model.predict(data, static_cam=True, postproc=postproc)
        if dev == "mps": torch.mps.synchronize()
        ts.append(time.time() - t0)
    glob = tonp(pred["body_params_global"])
    R["versions"][name] = {"t_first_s": ts[0], "t_median_s": float(np.median(ts)), "t_load_s": 0.0}
    return glob

ref = pytorch_version("pytorch_regression", data_full, postproc=False)
ref_j = C.fk_joints(soma, ref)
pp = pytorch_version("pytorch_postproc", data_full, postproc=True)
R["versions"]["pytorch_postproc"]["pa_mm_vs_ref"] = C.joint_metrics(C.fk_joints(soma, pp), ref_j)["pa_root_mpjpe_mm"]

t0 = time.time(); bvh = C.params_to_bvh(soma, pp, fps); R["stages"]["bvh_export_s"] = time.time() - t0
(C.RUNS / "bench" / f"{stem}_pytorch_postproc.bvh").write_text(bvh)
R["bvh_sanity"] = C.bvh_sanity(bvh.encode())
log("pytorch", {k: R["versions"][k] for k in ("pytorch_regression", "pytorch_postproc")}, "bvh", round(R["stages"]["bvh_export_s"], 2))

def onnx_version(name, path, ep, data):
    rec = {"model": str(path), "ep": ep}
    try:
        t0 = time.time(); sess = C.make_session(path, ep, log=lambda *s: None); rec["t_load_s"] = time.time() - t0
        rec["ep_active"] = sess.get_providers()[0]
        feed = feed_of(data); ts = []
        for _ in range(a.repeat):
            t0 = time.time(); out = C.run_sess(sess, feed); ts.append(time.time() - t0)
        rec["t_first_s"] = ts[0]; rec["t_median_s"] = float(np.median(ts))
        with torch.no_grad(): glob = tonp(decode_onnx(out, data))
        rec["finite"] = bool(all(np.isfinite(v).all() for v in glob.values()))
        rec["pa_mm_vs_ref"] = C.joint_metrics(C.fk_joints(soma, glob), ref_j)["pa_root_mpjpe_mm"]
        rec["mpjpe_mm_vs_ref"] = C.joint_metrics(C.fk_joints(soma, glob), ref_j)["mpjpe_mm"]
        del sess
    except Exception as e:
        rec["error"] = str(e)[:300]
    gc.collect(); R["versions"][name] = rec
    log(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k != "model"})
    return rec

Q = C.QUANT
onnx_version("onnx_fp32_coreml", C.MODELS["gem_denoiser"], "coreml", data_full)
onnx_version("onnx_fp32_cpu", C.MODELS["gem_denoiser"], "cpu", data_full)
onnx_version("onnx_fp16_cpu", Q / "gem_denoiser_fp16_ort.onnx", "cpu", data_full)
onnx_version("onnx_fp16_coreml", Q / "gem_denoiser_fp16_ort.onnx", "coreml", data_full)
onnx_version("onnx_int8pc_cpu", Q / "gem_denoiser_int8pc.onnx", "cpu", data_full)
onnx_version("onnx_w8_cpu", Q / "gem_denoiser_w8.onnx", "cpu", data_full)
onnx_version("noimg_fp32_cpu", Q / "gem_denoiser_no_imgfeat_consolidate.onnx", "cpu", data_noimg)
onnx_version("noimg_fp32_coreml", Q / "gem_denoiser_no_imgfeat_consolidate.onnx", "coreml", data_noimg)
onnx_version("noimg_fp16_cpu", Q / "gem_denoiser_no_imgfeat_fp16.onnx", "cpu", data_noimg)
onnx_version("noimg_int8_cpu", Q / "gem_denoiser_no_imgfeat_int8.onnx", "cpu", data_noimg)

del model; gc.collect(); demo._free_accelerator_memory()
if not a.skip_vitpose_int8:
    rec = {}
    try:
        frames = C.read_frames(video); bbx_xys = C.load_cache(cache)["bbx"]["bbx_xys"]
        t0 = time.time(); sess = C.make_session(Q / "vitpose_int8.onnx", "cpu", log=lambda *s: None); rec["t_load_s"] = time.time() - t0
        class Rn:
            input_names = [i.name for i in sess.get_inputs()]
            def __call__(self, **kw):
                feed = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}
                return {k: torch.from_numpy(v) for k, v in C.run_sess(sess, feed).items()}
        t0 = time.time(); kp = demo.run_vitpose_onnx(Rn(), "onnx", frames, bbx_xys); rec["t_run_s"] = time.time() - t0
        rec["ms_per_frame"] = rec["t_run_s"] / L * 1000
        kp_ref = data_noimg["kp2d"].numpy()
        rec["kp2d_px_mean_vs_mps"] = float(np.linalg.norm(kp.numpy()[..., :2] - kp_ref[..., :2], axis=-1).mean())
        rec["rss_mb"] = C.rss_mb()
        del sess, frames; gc.collect()
        d_edge = dict(data_noimg); d_edge["kp2d"] = kp.float()
        onnx_version("edge_vitpose_int8_noimg_fp16_cpu", Q / "gem_denoiser_no_imgfeat_fp16.onnx", "cpu", d_edge)
    except Exception as e:
        rec["error"] = str(e)[:300]
    R["stages"]["vitpose_int8"] = rec
    log("vitpose_int8", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})

S, V = R["stages"], R["versions"]
pre_noimg = S["preprocess_noimg_wall_s"]; pre_full = pre_noimg + S["sam3db_wall_s"]; bvh_t = S["bvh_export_s"]
def total(pre, v): return pre + V[v].get("t_load_s", 0) + V[v]["t_median_s"] + bvh_t if "t_median_s" in V[v] else None
R["totals_s"] = {
    "as_is_pytorch_postproc": total(pre_full, "pytorch_postproc"),
    "onnx_fp32_coreml": total(pre_full, "onnx_fp32_coreml"),
    "onnx_fp16_cpu": total(pre_full, "onnx_fp16_cpu"),
    "onnx_int8pc_cpu": total(pre_full, "onnx_int8pc_cpu"),
    "noimg_fp32_cpu": total(pre_noimg, "noimg_fp32_cpu"),
    "noimg_fp16_cpu": total(pre_noimg, "noimg_fp16_cpu"),
    "noimg_int8_cpu": total(pre_noimg, "noimg_int8_cpu"),
}
vp = S.get("vitpose_int8", {})
if "t_run_s" in vp and "t_median_s" in V.get("edge_vitpose_int8_noimg_fp16_cpu", {}):
    R["totals_s"]["edge_vitpose_int8_noimg_fp16"] = (S["Detection (YOLOX+ByteTrack)"] + S["read_video_s"] + vp["t_load_s"] + vp["t_run_s"]
                                                     + V["edge_vitpose_int8_noimg_fp16_cpu"]["t_load_s"] + V["edge_vitpose_int8_noimg_fp16_cpu"]["t_median_s"] + bvh_t)
R["peak_rss_mb"] = C.rss_mb()
C.save_json(C.RUNS / "bench" / f"{stem}.json", R)
log("totals", {k: round(v, 1) for k, v in R["totals_s"].items() if v is not None}, "peak_rss_mb", round(R["peak_rss_mb"]))
