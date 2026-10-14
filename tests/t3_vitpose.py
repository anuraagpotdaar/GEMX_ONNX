import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True); ap.add_argument("--video", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--frames", type=int, default=32); ap.add_argument("--out", required=True)
ap.add_argument("--model", default=None); ap.add_argument("--threads", type=int, default=None); ap.add_argument("--no-flip", action="store_true")
a = ap.parse_args()
demo = C.load_demo()
import torch
frames = C.read_frames(a.video)[: a.frames]
bbx_xys = C.load_cache(a.cache)["bbx"]["bbx_xys"][: a.frames]
N = len(frames)
t_load = time.time()
if a.backend.startswith("pytorch"):
    from gem.utils.vitpose_extractor import VitPoseExtractor
    dev = "mps" if a.backend.endswith("mps") else "cpu"
    vp = VitPoseExtractor(device=dev, pose_type="soma"); vp.flip_test = not a.no_flip
    t_load = time.time() - t_load; t0 = time.time()
    kp = vp.extract(frames, bbx_xys, img_ds=1.0, path_type="np")
    if dev == "mps": torch.mps.synchronize()
else:
    path = a.model or C.MODELS["vitpose"]
    sess = C.make_session(path, "coreml" if a.backend.endswith("coreml") else "cpu", threads=a.threads)
    class R:
        input_names = [i.name for i in sess.get_inputs()]
        def __call__(self, **kw):
            feed = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}
            return {k: torch.from_numpy(v) for k, v in C.run_sess(sess, feed).items()}
    t_load = time.time() - t_load; t0 = time.time()
    if a.no_flip:
        imgs, bb = demo.vitpose_preprocess(frames, bbx_xys); res = []
        for j in range(0, N, 16):
            hm = R()(imgs=imgs[j:j+16, :, :, 32:224])["heatmaps"].numpy(); res.append(demo.vitpose_postprocess(hm, bb[j:j+16]))
        kp = torch.cat(res)
    else:
        kp = demo.run_vitpose_onnx(R(), "onnx", frames, bbx_xys)
t_run = time.time() - t0
kp = kp.numpy()
np.savez(a.out, kp2d=kp, bbx_xys=bbx_xys.numpy(), t_load=t_load, t_run=t_run, n=N, backend=a.backend, rss_mb=C.rss_mb())
print(f"[t3_vitpose] {a.backend}: N={N} load={t_load:.1f}s run={t_run:.1f}s ({t_run/N*1000:.0f} ms/frame) rss={C.rss_mb():.0f}MB finite={np.isfinite(kp).all()} conf_mean={kp[...,2].mean():.3f}")
