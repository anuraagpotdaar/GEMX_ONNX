import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True); ap.add_argument("--video", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--frames", type=int, default=32); ap.add_argument("--out", required=True); ap.add_argument("--model", default=None)
a = ap.parse_args()
demo = C.load_demo(); demo._ensure_pipeline_deps()
import torch
from gem.utils.sam3db_extractor import SAM3DBExtractor
cache = C.load_cache(a.cache); bbx_xys = cache["bbx"]["bbx_xys"][: a.frames]
t0 = time.time()
ext = SAM3DBExtractor(device=demo.DEVICE)
if a.backend == "onnx-cpu":
    sess = C.make_session(a.model or C.MODELS["sam3db_backbone"], "cpu")
    class R:
        input_names = [i.name for i in sess.get_inputs()]
        def __call__(self, **kw):
            feed = {k: np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32) for k, v in kw.items()}
            return {k: torch.from_numpy(v) for k, v in C.run_sess(sess, feed).items()}
    model = ext.estimator.model
    model.backbone = demo._OnnxBackbone(R(), model.image_mean.clone(), model.image_std.clone()).to(demo.DEVICE)
t_load = time.time() - t0; t0 = time.time()
res = ext.extract_video_features(a.video, bbx_xys)
if demo.DEVICE == "mps": torch.mps.synchronize()
t_run = time.time() - t0
tok, tr = res["pose_tokens"].numpy(), res["transls"].numpy()

data = C.data_dict_from_cache(demo, a.video, a.cache, no_imgfeat=False)
N = len(tok); dsess = C.make_session(C.MODELS["gem_denoiser"], "cpu")
feed = {"obs": data["kp2d"][:N][None].numpy(), "bbx_xys": data["bbx_xys"][:N][None].numpy(), "K_fullimg": data["K_fullimg"][:N][None].numpy(),
        "f_imgseq": tok[None], "f_cam_angvel": data["cam_angvel"][:N][None].numpy()}
out = C.run_sess(dsess, {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in feed.items()})
np.savez(a.out, pose_tokens=tok, transls=tr, pred_x=out["pred_x"][0], pred_cam=out["pred_cam"][0], t_load=t_load, t_run=t_run, n=N, backend=a.backend, rss_mb=C.rss_mb())
print(f"[t3_sam3db] {a.backend}: N={N} load={t_load:.1f}s run={t_run:.1f}s ({t_run/N*1000:.0f} ms/frame) rss={C.rss_mb():.0f}MB tok_absmean={np.abs(tok).mean():.4f} finite={np.isfinite(tok).all()}")
