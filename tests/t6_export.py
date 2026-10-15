import argparse, gc, sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import common as C

ALL = ["pytorch_postproc", "onnx_fp32", "onnx_fp32_win120", "onnx_fp16", "onnx_int8pc", "noimg_fp16", "noimg_int8", "edge_vitpose_int8_noimg_fp16"]
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--versions", default=",".join(ALL)); ap.add_argument("--no-render", action="store_true")
a = ap.parse_args(); versions = a.versions.split(",")
video = Path(a.video).resolve(); stem = video.stem
demo = C.load_demo(); demo._ensure_pipeline_deps()
import torch, hydra, cv2
from omegaconf import open_dict
from gem.utils.video_io_utils import get_video_lwh
from gem.utils.cam_utils import compute_transl_full_cam
from gem.pipeline.gem_pipeline import get_body_params_w_Rt_v2
L, W, H = get_video_lwh(str(video)); fps = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) + 0.5) or 30
out_root = C.RUNS / "bench" / stem; cache = out_root / stem / "preprocess"; vis = C.RUNS / "visual" / stem; vis.mkdir(parents=True, exist_ok=True)
assert (cache / "vit_features.pt").exists(), f"no preprocess cache at {cache}; run t6_bench.py first"
cfg = C.build_cfg(demo, video, out_root, static_cam=True); dev = demo.DEVICE
data_full = C.data_dict_from_cache(demo, video, cache, no_imgfeat=False); data_noimg = C.data_dict_from_cache(demo, video, cache, no_imgfeat=True)
model = hydra.utils.instantiate(cfg.model, _recursive_=False); model.load_pretrained_model(cfg.ckpt_path or "inputs/pretrained/gem_soma.ckpt"); model = model.eval().to(dev)
ende = model.endecoder
if ende.obs_indices_dict is None: ende.build_obs_indices_dict()
soma = C.soma_layer()
tonp = lambda d: {k: v.detach().float().cpu() for k, v in d.items() if isinstance(v, torch.Tensor)}
def log(*s): print("[t6_export]", *s, flush=True)

def feed_of(data):
    f = {"obs": data["kp2d"][None].numpy(), "bbx_xys": data["bbx_xys"][None].numpy(), "K_fullimg": data["K_fullimg"][None].numpy(), "f_imgseq": data["f_imgseq"][None].numpy(), "f_cam_angvel": data["cam_angvel"][None].numpy()}
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in f.items()}

def pred_from_onnx(out, data):
    with torch.no_grad():
        pred_x = torch.from_numpy(out["pred_x"]).to(dev); pred_cam = torch.from_numpy(out["pred_cam"]).to(dev)
        dd = ende.decode(pred_x); sp = dd["scale_params"].clone(); sp[..., 0] = sp[..., 0].clamp(0.7, 1.0); dd["scale_params"] = sp
        incam = {"body_pose": dd["body_pose"][0], "global_orient": dd["global_orient"][0],
                 "transl": compute_transl_full_cam(pred_cam, data["bbx_xys"][None].to(dev), data["K_fullimg"][None].to(dev))[0],
                 "identity_coeffs": dd["identity_coeffs"][0], "scale_params": dd["scale_params"][0]}
        g = get_body_params_w_Rt_v2(global_orient_gv=dd["global_orient_gv"], local_transl_vel=dd["local_transl_vel"], global_orient_c=dd["global_orient"], cam_angvel=data["cam_angvel"][None].to(dev))
        glob = {"body_pose": dd["body_pose"][0], "global_orient": g["global_orient"][0], "transl": g["transl"][0], "identity_coeffs": dd["identity_coeffs"][0], "scale_params": dd["scale_params"][0]}
    return {"body_params_incam": tonp(incam), "body_params_global": tonp(glob), "K_fullimg": data["K_fullimg"]}

def run_onnx(path, data, window=None):
    sess = C.make_session(path, "cpu", log=lambda *s: None); feed = feed_of(data)
    if window is None: out = C.run_sess(sess, feed)
    else:
        px, pc = [], []
        for s in range(0, L, window):
            o = C.run_sess(sess, {k: np.ascontiguousarray(v[:, s:s + window]) for k, v in feed.items()}); px.append(o["pred_x"]); pc.append(o["pred_cam"])
        out = {"pred_x": np.concatenate(px, 1), "pred_cam": np.concatenate(pc, 1)}
    del sess; return pred_from_onnx(out, data)

Q = C.QUANT
def make_pred(v):
    if v == "pytorch_postproc":
        with torch.no_grad(): p = model.predict(data_full, static_cam=True, postproc=True)
        p = {"body_params_incam": tonp(p["body_params_incam"]), "body_params_global": tonp(p["body_params_global"]), "K_fullimg": data_full["K_fullimg"]}; return p
    if v == "onnx_fp32": return run_onnx(C.MODELS["gem_denoiser"], data_full)
    if v == "onnx_fp32_win120": return run_onnx(C.MODELS["gem_denoiser"], data_full, window=120)
    if v == "onnx_fp16": return run_onnx(Q / "gem_denoiser_fp16_ort.onnx", data_full)
    if v == "onnx_int8pc": return run_onnx(Q / "gem_denoiser_int8pc.onnx", data_full)
    if v == "noimg_fp16": return run_onnx(Q / "gem_denoiser_no_imgfeat_fp16.onnx", data_noimg)
    if v == "noimg_int8": return run_onnx(Q / "gem_denoiser_no_imgfeat_int8.onnx", data_noimg)
    if v == "edge_vitpose_int8_noimg_fp16":
        kp_path = vis / "vitpose_int8_kp2d.pt"
        if kp_path.exists(): kp = torch.load(kp_path)
        else:
            frames = C.read_frames(video); bbx_xys = C.load_cache(cache)["bbx"]["bbx_xys"]
            sess = C.make_session(Q / "vitpose_int8.onnx", "cpu", log=lambda *s: None)
            class Rn:
                input_names = [i.name for i in sess.get_inputs()]
                def __call__(self, **kw):
                    feed = {k: (x.cpu().numpy() if isinstance(x, torch.Tensor) else x) for k, x in kw.items()}
                    return {k: torch.from_numpy(x) for k, x in C.run_sess(sess, feed).items()}
            t0 = time.time(); kp = demo.run_vitpose_onnx(Rn(), "onnx", frames, bbx_xys).float(); log(f"vitpose int8 {time.time()-t0:.1f}s"); torch.save(kp, kp_path); del sess, frames; gc.collect()
        d = dict(data_noimg); d["kp2d"] = kp; return run_onnx(Q / "gem_denoiser_no_imgfeat_fp16.onnx", d)
    raise ValueError(v)

summary = {}
for v in versions:
    t0 = time.time(); pred = make_pred(v)
    bvh = C.params_to_bvh(soma, {k: x.numpy() for k, x in pred["body_params_global"].items()}, fps); (vis / f"{v}.bvh").write_text(bvh)
    torch.save(pred, vis / f"{v}_hpe_results.pt")
    rec = {"bvh": str(vis / f"{v}.bvh"), "t_pred_s": round(time.time() - t0, 2), "sanity": C.bvh_sanity(bvh.encode())}
    if not a.no_render:
        t0 = time.time()
        with open_dict(cfg):
            cfg.paths.hpe_results = str(vis / f"{v}_hpe_results.pt"); cfg.paths.incam_video = str(vis / f"{v}_incam.mp4")
            cfg.paths.global_video = str(vis / f"{v}_global.mp4"); cfg.paths.incam_global_horiz_video = str(vis / f"{v}_side.mp4")
        demo.render_incam(cfg, fps=fps); demo.render_global_o3d(cfg, fps=fps)
        demo.merge_videos_horizontal([cfg.paths.incam_video, cfg.paths.global_video], cfg.paths.incam_global_horiz_video)
        rec["t_render_s"] = round(time.time() - t0, 1); rec["side"] = str(vis / f"{v}_side.mp4")
    summary[v] = rec; log(v, rec); gc.collect()
(vis / "index.json").write_text(json.dumps({"video": str(video), "fps": fps, "frames": L, "versions": summary}, indent=2))
log("done", vis)
