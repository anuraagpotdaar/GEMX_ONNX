import argparse, json, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent.parent / "phone")); import gemx_phone as P

a = argparse.ArgumentParser(); a.add_argument("--clip", required=True); a.add_argument("--max_side", type=int, default=1080); a = a.parse_args()
root = Path(__file__).parent.parent; out_dir = root / "runs" / "phone" / a.clip
video = next(p for p in (Path("/Users/anuraag/Developer/GEM-X/sample_inputs") / f"{a.clip}{e}" for e in (".mp4", ".mov", ".MOV")) if p.exists())
cap = cv2.VideoCapture(str(video)); ok, bgr = cap.read(); cap.release(); assert ok
h, w = bgr.shape[:2]; s = min(1.0, a.max_side / max(h, w)); W, H = (int(round(w * s)), int(round(h * s))) if s < 1 else (w, h)
K = P.estimate_K(W, H)
r = np.load(out_dir / "phone_out.npz")
pp = P.PhonePipeline()
d = P.decode_x(r["pred_x"], pp.tab["x_mean"], pp.tab["x_std"])
rest = P.rest_skeleton(d["identity_coeffs"][0], d["scale_params"][0], pp.lut)
tr = P.incam_transl(r["pred_cam"], r["bbx_xys"], K)
joints = P.fk(rest, d["global_orient"], d["body_pose"], tr, pp.parents).astype(np.float32)
joints.tofile(out_dir / "joints_incam.bin"); K.astype(np.float32).tofile(out_dir / "K.bin"); r["pred_cam"].astype(np.float32).tofile(out_dir / "pred_cam.bin")

kp = r["kp2d"]; z = np.maximum(joints[..., 2], 1e-6)
u = K[0, 0] * joints[..., 0] / z + K[0, 2]; v = K[1, 1] * joints[..., 1] / z + K[1, 2]
err = np.sqrt((u - kp[..., 0]) ** 2 + (v - kp[..., 1]) ** 2); m = kp[..., 2] > 0.3
hm = [0, 67, 72, 1, 68, 73, 2, 69, 74, 3, 70, 75, 4, 11, 39, 6, 12, 40, 13, 41, 14, 42]
rep = {"clip": a.clip, "size": [W, H], "f": float(K[0, 0]), "frames": int(joints.shape[0]),
       "reproj_px_all_mean": float(err[m].mean()), "reproj_px_all_median": float(np.median(err[m])),
       "reproj_px_body22_mean": float(err[:, hm][m[:, hm]].mean()), "reproj_px_body22_p90": float(np.percentile(err[:, hm][m[:, hm]], 90)),
       "reproj_px_pelvis_mean": float(err[:, 0].mean()), "tz_m_min_max": [float(tr[:, 2].min()), float(tr[:, 2].max())],
       "rest_vs_rest_bin_max_mm": float(np.abs(rest - r["rest"]).max() * 1000)}
print(json.dumps(rep, indent=1)); (out_dir / "incam_oracle.json").write_text(json.dumps(rep, indent=1))
