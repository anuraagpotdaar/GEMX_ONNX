import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("ep", choices=["cpu", "coreml"])
ap.add_argument("--threads", type=int, default=None); ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--frames", type=int, default=32); ap.add_argument("--checker", action="store_true")
ap.add_argument("--tag", default="")
a = ap.parse_args()
C.chdir_gemx()
path = C.MODELS.get(a.model, Path(a.model))
name = Path(path).stem + (f"_{a.tag}" if a.tag else "")
rep = {"model": str(path), "ep": a.ep, "size_bytes": Path(path).stat().st_size +
       (Path(str(path) + ".data").stat().st_size if Path(str(path) + ".data").exists() else 0)}
if a.checker:
    import onnx
    t0 = time.time(); onnx.checker.check_model(str(path)); rep["checker_s"] = time.time() - t0
t0 = time.time()
try:
    sess = C.make_session(path, a.ep, threads=a.threads)
except Exception as e:
    rep["load_error"] = f"{type(e).__name__}: {e}"[:500]; C.save_json(C.REPORTS / f"t1_{name}_{a.ep}.json", rep); sys.exit(1)
rep["load_s"] = time.time() - t0; rep["providers"] = sess.get_providers(); rep["rss_after_load_mb"] = C.rss_mb()
rep["inputs"] = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
rep["outputs"] = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
rng = np.random.default_rng(0); B, L = a.batch, a.frames
stem = Path(path).stem
if stem.startswith("vitpose"):
    feed = {"imgs": rng.standard_normal((B, 3, 256, 192), dtype=np.float32)}
elif stem.startswith("sam3db"):
    feed = {"imgs": (rng.random((B, 3, 256, 256), dtype=np.float32) * 255)}
elif stem.startswith("gem_denoiser"):
    K = np.tile(np.array([[768, 0, 216], [0, 768, 384], [0, 0, 1]], np.float32), (1, L, 1, 1))
    feed = {"obs": np.concatenate([rng.random((1, L, 77, 2), dtype=np.float32) * 400, rng.random((1, L, 77, 1), dtype=np.float32)], -1),
            "bbx_xys": np.tile(np.array([216, 384, 500], np.float32), (1, L, 1)), "K_fullimg": K,
            "f_imgseq": rng.standard_normal((1, L, 1024), dtype=np.float32),
            "f_cam_angvel": np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (1, L, 1))}
elif stem.startswith("yolox"):
    feed = {"input": (rng.random((1, 3, 640, 640), dtype=np.float32) * 255)}
else:
    raise SystemExit(f"unknown model family {stem}")
try:
    t0 = time.time(); out = C.run_sess(sess, feed); rep["first_run_s"] = time.time() - t0
    ts = []
    for _ in range(3):
        t0 = time.time(); out = C.run_sess(sess, feed); ts.append(time.time() - t0)
    rep["run_s_median"] = float(np.median(ts)); rep["run_feed_shapes"] = {k: list(v.shape) for k, v in feed.items()}
    rep["out_shapes"] = {k: list(v.shape) for k, v in out.items()}
    rep["out_finite"] = {k: bool(np.isfinite(v).all()) for k, v in out.items()}
    rep["out_absmean"] = {k: float(np.abs(v).mean()) for k, v in out.items()}
except Exception as e:
    rep["run_error"] = f"{type(e).__name__}: {e}"[:800]
rep["rss_peak_mb"] = C.rss_mb()
C.save_json(C.REPORTS / f"t1_{name}_{a.ep}.json", rep)
print(json.dumps({k: v for k, v in rep.items() if k not in ("inputs", "outputs")}, indent=1, default=str))
sys.exit(1 if "run_error" in rep else 0)
