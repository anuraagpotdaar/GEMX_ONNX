from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
QUANT = ROOT / "quant"
DEFAULT_MODELS = {
    "yolox": Path.home() / ".cache/rtmlib/hub/checkpoints/yolox_x_8xb8-300e_humanart-a39d44ed.onnx",
    "vitpose": QUANT / "vitpose_int8.onnx",
    "vitpose_coreml": QUANT / "vitpose_w8.mlpackage",
    "denoiser": QUANT / "gem_denoiser_no_imgfeat_fp16_band.onnx",
    "tables": QUANT / "gemx_phone_tables.npz",
    "skeleton": QUANT / "soma_rest_skeleton_lut.npz",
}
IDX = {"body_pose": (0, 456), "identity_coeffs": (456, 501), "scale_params": (501, 570),
       "global_orient": (570, 576), "global_orient_gv": (576, 582), "local_transl_vel": (582, 585)}
VP_MEAN = np.array([0.485, 0.456, 0.406], np.float32); VP_STD = np.array([0.229, 0.224, 0.225], np.float32)

def make_session(path, threads=None):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads: so.intra_op_num_threads = threads
    return ort.InferenceSession(str(Path(path).resolve()), sess_options=so, providers=["CPUExecutionProvider"])

class CoreMLVitPose:
    def __init__(self, path, compute_units="ALL"):
        import coremltools as ct
        self.m = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, compute_units))
    def get_inputs(self): return []
    def run(self, names, feed):
        imgs = feed["imgs"]
        return [np.concatenate([self.m.predict({"imgs": np.ascontiguousarray(imgs[i:i + 1], dtype=np.float32)})["heatmaps"].astype(np.float32) for i in range(len(imgs))])]

def decode_video(path, max_side=1080):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; frames = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        h, w = bgr.shape[:2]; s = min(1.0, max_side / max(h, w))
        if s < 1.0: bgr = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        frames.append(bgr[..., ::-1])
    cap.release(); f = np.stack(frames); return f, float(fps), (f.shape[2], f.shape[1])

def estimate_K(width, height):
    f = float(max(width, height)); K = np.eye(3, dtype=np.float32); K[0, 0] = K[1, 1] = f; K[0, 2] = width / 2.0; K[1, 2] = height / 2.0; return K

def sample_indices(n, step):
    idx = np.arange(0, n, step)
    return idx if idx[-1] == n - 1 else np.append(idx, n - 1)

def _nms(boxes, scores, thr):
    order = scores.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]); b = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        order = order[1:][inter / (a + b - inter + 1e-9) <= thr]
    return keep

def yolox_detect(sess, rgb, score_thr=0.5, nms_thr=0.45, size=640):
    img = rgb[..., ::-1]; h, w = img.shape[:2]; r = min(size / h, size / w)
    resized = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, np.uint8); padded[:resized.shape[0], :resized.shape[1]] = resized
    out = sess.run(None, {sess.get_inputs()[0].name: padded.transpose(2, 0, 1)[None].astype(np.float32)})[0]
    if out.shape[-1] == 5:
        boxes, scores = out[0, :, :4] / r, out[0, :, 4]; m = scores > score_thr; return boxes[m], scores[m]
    grids, strides = [], []
    for s in (8, 16, 32):
        hs, ws = size // s, size // s; xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2)); strides.append(np.full((1, hs * ws, 1), s))
    grids, strides = np.concatenate(grids, 1), np.concatenate(strides, 1)
    out = out.copy(); out[..., :2] = (out[..., :2] + grids) * strides; out[..., 2:4] = np.exp(out[..., 2:4]) * strides
    p = out[0]; cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([(cx - bw / 2) / r, (cy - bh / 2) / r, (cx + bw / 2) / r, (cy + bh / 2) / r], 1)
    scores = p[:, 4] * p[:, 5]; m = scores > score_thr; boxes, scores = boxes[m], scores[m]
    keep = _nms(boxes, scores, nms_thr); return boxes[keep], scores[keep]

def primary_boxes(frames, idx, sess, W, H):
    boxes = np.full((len(idx), 4), np.nan, np.float32)
    for k, i in enumerate(idx):
        b, s = yolox_detect(sess, frames[i])
        if len(b): boxes[k] = b[((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax()]
    valid = ~np.isnan(boxes[:, 0])
    if not valid.any(): return np.tile(np.array([0, 0, W - 1, H - 1], np.float32), (len(idx), 1))
    v = np.where(valid)[0]
    for c in range(4): boxes[:, c] = np.interp(np.arange(len(idx)), v, boxes[v, c])
    return boxes

def smooth_xyxy(b, window=3):
    if len(b) <= window: return b
    pad = window // 2; bp = np.concatenate([np.repeat(b[:1], pad, 0), b, np.repeat(b[-1:], pad, 0)])
    return np.stack([np.convolve(bp[:, c], np.ones(window) / window, mode="valid") for c in range(4)], 1).astype(np.float32)

def xyxy_to_xys(b, W, H, enlarge=1.2):
    b = b.copy(); b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, W - 1); b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, H - 1)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2; w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    ar = 192 / 256; h = np.where(w > ar * h, w / ar, h); w = np.where(w < ar * h, h * ar, w)
    return np.stack([cx, cy, np.maximum(h, w) * enlarge], 1).astype(np.float32)

def vitpose_crop(frame_rgb, cx, cy, s):
    hs = s / 2; src = np.array([[cx - hs, cy - hs], [cx + hs, cy - hs], [cx, cy]], np.float32); dst = np.array([[0, 0], [255, 0], [127.5, 127.5]], np.float32)
    crop = cv2.warpAffine(frame_rgb, cv2.getAffineTransform(src, dst), (256, 256), flags=cv2.INTER_LINEAR)
    crop = (crop[..., ::-1].astype(np.float32) / 255.0 - VP_MEAN) / VP_STD
    return crop.transpose(2, 0, 1)[:, :, 32:224]

def heatmaps_to_kp2d(hm, bbx_xys):
    B, J, H, W = hm.shape; flat = hm.reshape(B, J, -1); idx = flat.argmax(-1); px = (idx % W).astype(np.float32); py = (idx // W).astype(np.float32)
    for n in range(B):
        for k in range(J):
            x, y = int(px[n, k]), int(py[n, k]); h = hm[n, k]
            if 1 < x < W - 1: px[n, k] += np.sign(h[y, x + 1] - h[y, x - 1]) * 0.25
            if 1 < y < H - 1: py[n, k] += np.sign(h[y + 1, x] - h[y - 1, x]) * 0.25
    s = bbx_xys[:, 2:3]; sw, sh = s * 24 / 32, s
    x = px / W * sw + bbx_xys[:, 0:1] - sw / 2; y = py / H * sh + bbx_xys[:, 1:2] - sh / 2
    return np.stack([x, y, flat.max(-1)], -1).astype(np.float32)

FLIP_PAIRS = [(9, 10)] + [(k, k + 28) for k in range(11, 39)] + [(k, k + 5) for k in range(67, 72)]

def flip_heatmaps(hm):
    y = hm.copy()
    for l, r in FLIP_PAIRS: y[:, l], y[:, r] = hm[:, r], hm[:, l]
    return y[:, :, :, ::-1]

def vitpose_run(sess, frames, idx, bbx_xys_s, batch=8, flip_test=False):
    out = []
    for j in range(0, len(idx), batch):
        ii = idx[j:j + batch]; crops = np.stack([vitpose_crop(frames[i], *bbx_xys_s[j + k]) for k, i in enumerate(ii)])
        hm = sess.run(["heatmaps"], {"imgs": crops})[0]
        if flip_test: hm = 0.5 * (hm + flip_heatmaps(sess.run(["heatmaps"], {"imgs": np.ascontiguousarray(crops[:, :, :, ::-1])})[0]))
        out.append(heatmaps_to_kp2d(hm, bbx_xys_s[j:j + batch]))
    return np.concatenate(out)

def interp_frames(x_s, idx, n):
    flat = x_s.reshape(len(idx), -1); t = np.arange(n)
    return np.stack([np.interp(t, idx, flat[:, c]) for c in range(flat.shape[1])], 1).reshape((n,) + x_s.shape[1:]).astype(np.float32)

def band_mask(L, max_len=120):
    m = np.zeros((L, L), bool)
    if L <= max_len: return m
    m[:] = True; eff = min(max_len, L)
    for i in range(L):
        lo, hi = max(0, i - max_len // 2), min(L, i + max_len // 2); m[i, min(L - eff, lo):max(eff, hi)] = False
    return m

DENOISE_WINDOW, DENOISE_OVERLAP = 900, 180

def denoise(sess, kp2d, bbx_xys, K, cam_angvel_static, window=120):
    L = len(kp2d); has_mask = any(i.name == "attn_mask" for i in sess.get_inputs())
    if has_mask and L > DENOISE_WINDOW:
        px, pc = np.zeros((L, 585), np.float32), np.zeros((L, 3), np.float32); s = 0
        while True:
            e = min(s + DENOISE_WINDOW, L); o = denoise(sess, kp2d[s:e], bbx_xys[s:e], K, cam_angvel_static, window)
            ks, ke = (0 if s == 0 else s + DENOISE_OVERLAP // 2), (L if e == L else e - DENOISE_OVERLAP // 2)
            px[ks:ke], pc[ks:ke] = o[0][ks - s:ke - s], o[1][ks - s:ke - s]
            if e == L: return px, pc
            s = e - DENOISE_OVERLAP
    bounds = np.array([0, L]) if has_mask else np.linspace(0, L, int(np.ceil(L / window)) + 1).astype(int); px, pc = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        feed = {"obs": kp2d[None, a:b], "bbx_xys": bbx_xys[None, a:b], "K_fullimg": np.repeat(K[None, None], b - a, 1),
                "f_imgseq": np.zeros((1, b - a, 1024), np.float32), "f_cam_angvel": np.repeat(cam_angvel_static[None, None], b - a, 1)}
        feed = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in feed.items()}
        if has_mask: feed["attn_mask"] = band_mask(b - a, window)
        o = sess.run(["pred_x", "pred_cam"], feed); px.append(o[0][0]); pc.append(o[1][0])
    return np.concatenate(px), np.concatenate(pc)

def r6d_to_rotvec(d6):
    a1, a2 = d6[..., :3], d6[..., 3:6]; b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1; b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True); b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], -2)
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(d6.shape[:-1] + (3,)).astype(np.float32)

def decode_x(pred_x, x_mean, x_std):
    x = pred_x.astype(np.float64) * x_std + x_mean; g = lambda k: x[:, IDX[k][0]:IDX[k][1]]
    scale = g("scale_params").copy(); scale[:, 0] = np.clip(scale[:, 0], 0.7, 1.0)
    return {"body_pose": r6d_to_rotvec(g("body_pose").reshape(len(x), 76, 6)).reshape(len(x), 228), "identity_coeffs": g("identity_coeffs").astype(np.float32),
            "scale_params": scale.astype(np.float32), "global_orient": r6d_to_rotvec(g("global_orient")), "global_orient_gv": r6d_to_rotvec(g("global_orient_gv")),
            "local_transl_vel": g("local_transl_vel").astype(np.float32)}

def global_frame_static(global_orient_gv, local_transl_vel):
    go = global_orient_gv.copy(); go[np.linalg.norm(go, axis=-1) < 1e-5] = 0
    v = Rotation.from_rotvec(go).apply(local_transl_vel)
    transl = np.cumsum(np.concatenate([np.zeros((1, 3)), v[:-1]]), 0)
    return go.astype(np.float32), transl.astype(np.float32)

def incam_transl(pred_cam, bbx_xys, K):
    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]; f, icx, icy = K[0, 0], K[0, 2], K[1, 2]; sb = s * bbx_xys[:, 2]
    return np.stack([tx + 2 * (bbx_xys[:, 0] - icx) / (sb + 1e-9), ty + 2 * (bbx_xys[:, 1] - icy) / (sb + 1e-9), 2 * f / (sb + 1e-9)], 1).astype(np.float32)

def rest_skeleton(identity, scale, lut):
    J1, A, grid, L = lut["J1"], lut["A"], lut["grid"], lut["LUT"]; out = J1 + np.einsum("i,ijk->jk", identity, A); d = grid[1] - grid[0]
    for k in range(1, len(scale)):
        v = np.clip(scale[k], grid[0], grid[-1]); gi = min(int((v - grid[0]) / d), len(grid) - 2); t = (v - grid[gi]) / d
        out = out + (1 - t) * L[k, gi] + t * L[k, gi + 1]
    return (scale[0] * out).astype(np.float32)

def fk(rest, global_orient, body_pose, transl, parents78):
    L = len(transl); rot = np.concatenate([global_orient[:, None], body_pose.reshape(L, 76, 3)], 1)
    Rl = Rotation.from_rotvec(rot.reshape(-1, 3)).as_matrix().reshape(L, 77, 3, 3); G = np.zeros_like(Rl); P = np.zeros((L, 77, 3))
    par = parents78[1:] - 1
    G[:, 0] = Rl[:, 0]; P[:, 0] = transl
    for k in range(1, 77):
        q = par[k]; G[:, k] = G[:, q] @ Rl[:, k]; P[:, k] = P[:, q] + np.einsum("lij,j->li", G[:, q], rest[k] - rest[q])
    return P.astype(np.float32)

def ground(transl, joints, smooth_xz=True, sigma=3):
    t = transl.astype(np.float64).copy()
    if smooth_xz and len(t) > 1:
        t[:, 0] = gaussian_filter1d(t[:, 0], sigma, mode="nearest", truncate=4.0); t[:, 2] = gaussian_filter1d(t[:, 2], sigma, mode="nearest", truncate=4.0)
    j = joints - transl[:, None] + t[:, None]; t[:, 1] -= j[..., 1].min()
    return t.astype(np.float32)

def to_bvh(global_orient, body_pose, transl, rest, names78, parents78, fps):
    n = 78; pos = np.concatenate([np.zeros((1, 3)), rest]); off = np.zeros((n, 3))
    for j in range(1, n): off[j] = pos[j] - pos[parents78[j]]
    L = len(transl); rotvecs = np.concatenate([global_orient[:, None], body_pose.reshape(L, 76, 3)], 1); euler = np.zeros((L, n, 3))
    for j in range(1, n): euler[:, j] = Rotation.from_rotvec(rotvecs[:, j - 1]).as_euler("ZYX", degrees=True)
    children = [[] for _ in range(n)]
    for j in range(2, n): children[parents78[j]].append(j)
    out, order = ["HIERARCHY"], []
    def w(j, d):
        ind = "\t" * d; root = j == 1; order.append(j); out.append(f"{ind}{'ROOT' if root else 'JOINT'} {names78[j]}"); out.append(f"{ind}{{")
        o = np.zeros(3) if root else off[j]; out.append(f"{ind}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
        out.append(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation" if root else f"{ind}\tCHANNELS 3 Zrotation Yrotation Xrotation")
        if children[j]:
            for c in children[j]: w(c, d + 1)
        else: out.extend([f"{ind}\tEnd Site", f"{ind}\t{{", f"{ind}\t\tOFFSET 0.000000 0.000000 0.000000", f"{ind}\t}}"])
        out.append(f"{ind}}}")
    w(1, 0); out += ["MOTION", f"Frames: {L}", f"Frame Time: {1.0 / (fps if fps > 0 else 30.0):.6f}"]
    motion = np.concatenate([transl, euler[:, order].reshape(L, -1)], 1); out += [" ".join(f"{v:.6f}" for v in row) for row in motion]
    return "\n".join(out) + "\n"

class PhonePipeline:
    def __init__(self, models=None, threads=None, vitpose="coreml", flip_test=True):
        m = dict(DEFAULT_MODELS); m.update(models or {}); self.m = m; self.vitpose_backend = vitpose; self.flip_test = flip_test
        self.tab = dict(np.load(m["tables"])); self.lut = dict(np.load(m["skeleton"]))
        self.names = [str(x) for x in self.tab["joint_names"]]; self.parents = self.tab["joint_parents"].astype(int)
        self.det = self.vp = self.den = None; self.threads = threads

    def load(self):
        t0 = time.time(); self.det = make_session(self.m["yolox"], self.threads); self.den = make_session(self.m["denoiser"], self.threads)
        self.vp = CoreMLVitPose(self.m["vitpose_coreml"]) if self.vitpose_backend == "coreml" else make_session(self.m["vitpose"], self.threads)
        return time.time() - t0

    def run(self, video, step=3, window=120, max_side=1080, smooth_xz=True, frames=None, fps=None, bbx_xys_full=None, kp2d_full=None, size=None, resample_fps=30):
        T = {}; t0 = time.time()
        if frames is None and kp2d_full is None: frames, fps, _ = decode_video(video, max_side)
        T["decode"] = time.time() - t0; N = len(frames) if frames is not None else len(kp2d_full); W, H = size if size else (frames.shape[2], frames.shape[1]); K = estimate_K(W, H)
        resampling = resample_fps is not None and fps and abs(fps - resample_fps) >= 1
        step_eff = max(1, int(round(step * fps / 30))) if resampling else step; idx = sample_indices(N, step_eff)
        n_out, pos = (int(round((N - 1) * resample_fps / fps)) + 1, idx * resample_fps / fps) if resampling else (N, idx)
        if resampling: fps = resample_fps
        if kp2d_full is None:
            t0 = time.time(); xyxy = smooth_xyxy(primary_boxes(frames, idx, self.det, W, H)); bbx_s = xyxy_to_xys(xyxy, W, H); T["detect"] = time.time() - t0
            t0 = time.time(); kp_s = vitpose_run(self.vp, frames, idx, bbx_s, flip_test=self.flip_test); T["vitpose"] = time.time() - t0
            kp2d, bbx_xys = interp_frames(kp_s, pos, n_out), interp_frames(bbx_s, pos, n_out)
        else: kp2d, bbx_xys = kp2d_full, bbx_xys_full
        t0 = time.time(); pred_x, pred_cam = denoise(self.den, kp2d, bbx_xys, K, self.tab["cam_angvel_static"], window); T["denoise"] = time.time() - t0
        t0 = time.time(); d = decode_x(pred_x, self.tab["x_mean"], self.tab["x_std"]); go, tr = global_frame_static(d["global_orient_gv"], d["local_transl_vel"])
        rest = rest_skeleton(d["identity_coeffs"][0], d["scale_params"][0], self.lut); joints = fk(rest, go, d["body_pose"], tr, self.parents)
        tr_g = ground(tr, joints, smooth_xz); joints_g = joints - tr[:, None] + tr_g[:, None]
        bvh = to_bvh(go, d["body_pose"], tr_g, rest, self.names, self.parents, fps); T["post+bvh"] = time.time() - t0
        return {"bvh": bvh, "joints": joints_g, "rest": rest, "params": {"body_pose": d["body_pose"], "global_orient": go, "transl": tr_g, "identity_coeffs": d["identity_coeffs"], "scale_params": d["scale_params"]},
                "incam": {"global_orient": d["global_orient"], "transl": incam_transl(pred_cam, bbx_xys, K)}, "kp2d": kp2d, "bbx_xys": bbx_xys, "pred_x": pred_x, "pred_cam": pred_cam,
                "fps": fps, "frames": N, "size": (W, H), "sampled": len(idx), "timings": T}

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--out", default=None); ap.add_argument("--step", type=int, default=3); ap.add_argument("--threads", type=int, default=None); ap.add_argument("--max-side", type=int, default=1080); ap.add_argument("--native-fps", action="store_true", help="keep the clip's frame timeline instead of resampling to 30 fps")
    ap.add_argument("--vitpose", default="coreml", choices=["coreml", "onnx"]); ap.add_argument("--no-flip", action="store_true", help="single VitPose pass (no flip test)")
    a = ap.parse_args(); p = PhonePipeline(threads=a.threads, vitpose=a.vitpose, flip_test=not a.no_flip); t_load = p.load(); r = p.run(a.video, step=a.step, max_side=a.max_side, resample_fps=None if a.native_fps else 30)
    out = Path(a.out or Path(a.video).with_suffix(".phone.bvh")); out.write_text(r["bvh"])
    print(json.dumps({"out": str(out), "frames": r["frames"], "sampled": r["sampled"], "size": r["size"], "fps": r["fps"], "t_load": round(t_load, 2), "timings": {k: round(v, 2) for k, v in r["timings"].items()}}))
