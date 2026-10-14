import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import common as C

ap = argparse.ArgumentParser(); ap.add_argument("kind", choices=["vitpose", "params"]); ap.add_argument("files", nargs="+")
ap.add_argument("--fps", type=float, default=10); ap.add_argument("--bvh", action="store_true"); ap.add_argument("--report", default=None)
a = ap.parse_args()
ref = np.load(a.files[0], allow_pickle=True); results = {}
if a.kind == "vitpose":
    k0 = ref["kp2d"]; print(f"ref {a.files[0]} backend={ref['backend']} n={ref['n']} {float(ref['t_run'])/int(ref['n'])*1000:.0f} ms/frame")
    for f in a.files[1:]:
        o = np.load(f, allow_pickle=True); k = o["kp2d"]; n = min(len(k0), len(k))
        d = np.linalg.norm(k0[:n, :, :2] - k[:n, :, :2], axis=-1); conf = (k0[:n, :, 2] > 0.5) & (k[:n, :, 2] > 0.5)
        s = (k0[:n, :, 2] + k[:n, :, 2]) / 2
        r = {"backend": str(o["backend"]), "n": n, "ms_per_frame": float(o["t_run"]) / int(o["n"]) * 1000, "load_s": float(o["t_load"]),
             "rss_mb": float(o["rss_mb"]), "px_mean": float(d.mean()), "px_mean_conf>0.5": float(d[conf].mean()) if conf.any() else None,
             "px_p95": float(np.percentile(d, 95)), "px_max": float(d.max()), "conf_absdiff_mean": float(np.abs(k0[:n, :, 2] - k[:n, :, 2]).mean()),
             "bbox_size_mean_px": float(o["bbx_xys"][:n, 2].mean())}
        r["px_mean_rel_bbox_pct"] = r["px_mean"] / r["bbox_size_mean_px"] * 100
        results[Path(f).name] = r; print(Path(f).name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
else:
    soma = C.soma_layer()
    def params(z, which):
        return {k[len(which) + 1:]: z[k] for k in z.files if k.startswith(which + "_")}
    gref = params(ref, "global"); jref = C.fk_joints(soma, gref); print(f"ref {a.files[0]} backend={ref['backend']} L={len(jref)} t_run={float(ref['t_run_s']):.2f}s")
    for f in a.files[1:]:
        o = np.load(f, allow_pickle=True); g = params(o, "global"); ic = params(o, "incam"); icr = params(ref, "incam")
        j = C.fk_joints(soma, g); r = {"backend": str(o["backend"]), "t_run_s": float(o["t_run_s"])}
        r["global_joints"] = C.joint_metrics(jref, j)
        r["incam_joints"] = C.joint_metrics(C.fk_joints(soma, icr), C.fk_joints(soma, ic))
        r["body_pose"] = C.rotvec_angle_deg(gref["body_pose"], g["body_pose"])
        r["body_pose_body_only(j1-10,67-76)"] = C.rotvec_angle_deg(gref["body_pose"].reshape(-1, 76, 3)[:, [*range(0, 10), *range(66, 76)]], g["body_pose"].reshape(-1, 76, 3)[:, [*range(0, 10), *range(66, 76)]])
        r["global_orient"] = C.rotvec_angle_deg(gref["global_orient"], g["global_orient"])
        r["transl_mm"] = float(np.linalg.norm(gref["transl"] - g["transl"], axis=-1).mean() * 1000)
        r["identity_absdiff"] = float(np.abs(gref["identity_coeffs"] - g["identity_coeffs"]).mean()); r["scale_absdiff"] = float(np.abs(gref["scale_params"] - g["scale_params"]).mean())
        if "pred_x" in ref.files and "pred_x" in o.files: r["pred_x_maxabs"] = float(np.abs(ref["pred_x"] - o["pred_x"]).max()); r["pred_x_meanabs"] = float(np.abs(ref["pred_x"] - o["pred_x"]).mean())
        if a.bvh:
            bvh = C.params_to_bvh(soma, g, a.fps); names, pos, ft = C.bvh_fk(bvh)
            keep = [i for i, n in enumerate(names) if not n.endswith("_End")]
            r["bvh_fk_vs_soma_joints_mm"] = C.joint_metrics(j, pos[:, keep]); r["bvh_sanity"] = C.bvh_sanity(bvh.encode()); r["bvh_bytes"] = len(bvh)
            (C.RUNS / "bvh").mkdir(exist_ok=True); (C.RUNS / "bvh" / (Path(f).stem + ".bvh")).write_text(bvh)
        results[Path(f).name] = r
        import json; print(Path(f).name, json.dumps(r, indent=1))
if a.report: C.save_json(a.report, {"ref": a.files[0], "results": results})
