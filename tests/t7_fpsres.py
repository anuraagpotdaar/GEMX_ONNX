import argparse, sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import common as C
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--ref-fps", type=float, required=True); ap.add_argument("--no-render", action="store_true")
a = ap.parse_args(); video = Path(a.video).resolve(); stem = video.stem
demo = C.load_demo(); demo._ensure_pipeline_deps()
import torch, hydra, cv2
from omegaconf import open_dict
from gem.utils.video_io_utils import get_video_lwh
L, W, H = get_video_lwh(str(video)); fps = float(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)) or 30.0
out_root = C.RUNS / "fpsres" / stem; cfg = C.build_cfg(demo, video, out_root, static_cam=True); vis = out_root; vis.mkdir(parents=True, exist_ok=True)
demo._timings.clear(); t0 = time.time(); demo.run_preprocess_fast(cfg, no_imgfeat=False); t_pre = time.time() - t0
bbx = torch.load(out_root / stem / "preprocess" / "bbx.pt")["bbx_xys"]
data = C.data_dict_from_cache(demo, video, out_root / stem / "preprocess", no_imgfeat=False)
model = hydra.utils.instantiate(cfg.model, _recursive_=False); model.load_pretrained_model(cfg.ckpt_path or "inputs/pretrained/gem_soma.ckpt"); model = model.eval().to(demo.DEVICE)
tonp = lambda d: {k: v.detach().float().cpu() for k, v in d.items() if isinstance(v, torch.Tensor)}
t0 = time.time()
with torch.no_grad(): p = model.predict(data, static_cam=True, postproc=True)
t_den = time.time() - t0
pred = {"body_params_incam": tonp(p["body_params_incam"]), "body_params_global": tonp(p["body_params_global"]), "K_fullimg": data["K_fullimg"]}
torch.save(pred, vis / "pytorch_postproc_hpe_results.pt")
soma = C.soma_layer(); bvh = C.params_to_bvh(soma, {k: v.numpy() for k, v in pred["body_params_global"].items()}, int(round(fps))); (vis / "pytorch_postproc.bvh").write_text(bvh)

ref = torch.load(a.ref, weights_only=False); jr = C.fk_joints(soma, {k: v.numpy() for k, v in ref["body_params_global"].items()}); jv = C.fk_joints(soma, {k: v.numpy() for k, v in pred["body_params_global"].items()})
idx = np.clip(np.round(np.arange(L) * a.ref_fps / fps).astype(int), 0, len(jr) - 1)
m = C.joint_metrics(jv, jr[idx]); m_incam = C.joint_metrics(C.fk_joints(soma, {k: v.numpy() for k, v in pred["body_params_incam"].items()}), C.fk_joints(soma, {k: v.numpy() for k, v in ref["body_params_incam"].items()})[idx], root_align=True)

def height(params):
    r = soma.get_skeleton(torch.as_tensor(params["identity_coeffs"][:1]).float(), torch.as_tensor(params["scale_params"][:1]).float())[0]
    return float(r[:, 1].max() - r[:, 1].min())
rep = {"video": str(video), "frames": L, "w": W, "h": H, "fps": fps, "bbox_px_mean": float(bbx[:, 2].mean()), "t_preprocess_s": t_pre, "t_denoiser_s": t_den,
       "stages": {k: float(v) for k, v in demo._timings.items()}, "vs_ref_global": m, "vs_ref_incam": m_incam,
       "height_m": height({k: v.numpy() for k, v in pred["body_params_global"].items()}), "ref_height_m": height({k: v.numpy() for k, v in ref["body_params_global"].items()}),
       "sanity": C.bvh_sanity(bvh.encode())}
if not a.no_render:
    with open_dict(cfg):
        cfg.paths.hpe_results = str(vis / "pytorch_postproc_hpe_results.pt"); cfg.paths.incam_video = str(vis / "pytorch_postproc_incam.mp4")
        cfg.paths.global_video = str(vis / "pytorch_postproc_global.mp4"); cfg.paths.incam_global_horiz_video = str(vis / "pytorch_postproc_side.mp4")
    demo.render_incam(cfg, fps=int(round(fps))); demo.render_global_o3d(cfg, fps=int(round(fps))); demo.merge_videos_horizontal([cfg.paths.incam_video, cfg.paths.global_video], cfg.paths.incam_global_horiz_video)
C.save_json(vis / "report.json", rep); print("[t7]", json.dumps(rep, default=float))
