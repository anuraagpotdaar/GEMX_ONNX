import sys, numpy as np; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import common as C
ref = np.load(sys.argv[1]); res = {}
for f in sys.argv[2:]:
    o = np.load(f); n = min(int(ref["n"]), int(o["n"])); a, b = ref["pose_tokens"][:n], o["pose_tokens"][:n]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    r = {"backend": str(o["backend"]), "ms_per_frame": float(o["t_run"]) / int(o["n"]) * 1000, "ref_ms_per_frame": float(ref["t_run"]) / int(ref["n"]) * 1000,
         "rss_mb": float(o["rss_mb"]), "token_cos_mean": float(cos.mean()), "token_cos_min": float(cos.min()), "token_absdiff_mean": float(np.abs(a - b).mean()), "token_absmean": float(np.abs(a).mean()),
         "transl_mm": float(np.linalg.norm(ref["transls"][:n] - o["transls"][:n], axis=-1).mean() * 1000),
         "pred_x_meanabs": float(np.abs(ref["pred_x"][:n] - o["pred_x"][:n]).mean()), "pred_x_maxabs": float(np.abs(ref["pred_x"][:n] - o["pred_x"][:n]).max()),
         "pred_x_absmean": float(np.abs(ref["pred_x"][:n]).mean())}
    res[Path(f).name] = r; print(Path(f).name, r)
C.save_json(C.REPORTS / "t3_sam3db_parity.json", {"ref": sys.argv[1], "results": res})
