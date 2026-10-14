import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True); ap.add_argument("--video", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--no_imgfeat", action="store_true"); ap.add_argument("--model", default=None)
ap.add_argument("--threads", type=int, default=None); ap.add_argument("--repeat", type=int, default=3); ap.add_argument("--device", default=None)
a = ap.parse_args()
demo = C.load_demo(); demo._ensure_pipeline_deps()
if a.device: demo.DEVICE = a.device
import torch, hydra
data = C.data_dict_from_cache(demo, a.video, a.cache, no_imgfeat=a.no_imgfeat, static_cam=True)
cfg = C.build_cfg(demo, a.video, C.RUNS / "_cfg_tmp", static_cam=True)
L = int(data["length"])
model = hydra.utils.instantiate(cfg.model, _recursive_=False)
model.load_pretrained_model(cfg.ckpt_path or "inputs/pretrained/gem_soma.ckpt")
dev = demo.DEVICE
model = model.eval().to(dev)
ende = model.endecoder
if ende.obs_indices_dict is None: ende.build_obs_indices_dict()
rep = {"backend": a.backend, "L": L, "no_imgfeat": a.no_imgfeat}
with torch.no_grad():
    if a.backend.startswith("onnx"):
        path = a.model or C.MODELS["gem_denoiser_no_imgfeat" if a.no_imgfeat else "gem_denoiser"]
        sess = C.make_session(path, "coreml" if a.backend.endswith("coreml") else "cpu", threads=a.threads)
        feed = {"obs": data["kp2d"][None].numpy(), "bbx_xys": data["bbx_xys"][None].numpy(), "K_fullimg": data["K_fullimg"][None].numpy(),
                "f_imgseq": data["f_imgseq"][None].numpy(), "f_cam_angvel": data["cam_angvel"][None].numpy()}
        feed = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in feed.items()}
        ts = []
        for _ in range(a.repeat):
            t0 = time.time(); out = C.run_sess(sess, feed); ts.append(time.time() - t0)
        rep["t_run_s"] = float(np.median(ts)); rep["t_first_s"] = ts[0]
        pred_x = torch.from_numpy(out["pred_x"]).to(dev); pred_cam = torch.from_numpy(out["pred_cam"]).to(dev)
        dd = ende.decode(pred_x)
        sp = dd["scale_params"].clone(); sp[..., 0] = sp[..., 0].clamp(0.7, 1.0); dd["scale_params"] = sp
        from gem.utils.cam_utils import compute_transl_full_cam
        from gem.pipeline.gem_pipeline import get_body_params_w_Rt_v2
        incam = {"body_pose": dd["body_pose"][0], "global_orient": dd["global_orient"][0],
                 "transl": compute_transl_full_cam(pred_cam, data["bbx_xys"][None].to(dev), data["K_fullimg"][None].to(dev))[0],
                 "identity_coeffs": dd["identity_coeffs"][0], "scale_params": dd["scale_params"][0]}
        g = get_body_params_w_Rt_v2(global_orient_gv=dd["global_orient_gv"], local_transl_vel=dd["local_transl_vel"],
                                    global_orient_c=dd["global_orient"], cam_angvel=data["cam_angvel"][None].to(dev))
        glob = {"body_pose": dd["body_pose"][0], "global_orient": g["global_orient"][0], "transl": g["transl"][0],
                "identity_coeffs": dd["identity_coeffs"][0], "scale_params": dd["scale_params"][0]}
        extra = {"pred_x": out["pred_x"][0], "pred_cam": out["pred_cam"][0]}
    else:
        if a.backend == "pytorch-ddim": model.pipeline.regression_only = False
        if a.backend == "pytorch-ddim-fixed": model.pipeline.denoiser3d.regression_only = False
        postproc = a.backend in ("pytorch-postproc", "pytorch-ddim", "pytorch-ddim-fixed")
        ts = []
        for _ in range(a.repeat if a.backend == "pytorch" else 1):
            t0 = time.time(); pred = model.predict(data, static_cam=True, postproc=postproc)
            if dev == "mps": torch.mps.synchronize()
            ts.append(time.time() - t0)
        rep["t_run_s"] = float(np.median(ts)); rep["t_first_s"] = ts[0]
        incam, glob = pred["body_params_incam"], pred["body_params_global"]; extra = {}
tonp = lambda d: {k: v.detach().float().cpu().numpy() for k, v in d.items() if isinstance(v, torch.Tensor)}
np.savez(a.out, **{f"incam_{k}": v for k, v in tonp(incam).items()}, **{f"global_{k}": v for k, v in tonp(glob).items()}, **extra, **rep)
rep["rss_mb"] = C.rss_mb(); rep["finite"] = all(np.isfinite(v).all() for v in tonp(glob).values())
print("[t3_denoiser]", rep)
