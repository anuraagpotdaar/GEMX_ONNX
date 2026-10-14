from __future__ import annotations

import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

GEMX_ROOT = Path(os.environ.get("GEMX_ROOT", "/Users/anuraag/Developer/GEM-X"))
WORK = Path(os.environ.get("GEMX_WORK", "/Users/anuraag/Developer/GEMX_ONNX"))
RUNS, QUANT, REPORTS = WORK / "runs", WORK / "quant", WORK / "reports"
for d in (RUNS, QUANT, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

ONNX_DIR = GEMX_ROOT / "inputs" / "onnx"
YOLOX_ONNX = Path.home() / ".cache/rtmlib/hub/checkpoints/yolox_x_8xb8-300e_humanart-a39d44ed.onnx"
MODELS = {
    "vitpose": ONNX_DIR / "vitpose.onnx",
    "sam3db_backbone": ONNX_DIR / "sam3db_backbone.onnx",
    "gem_denoiser": ONNX_DIR / "gem_denoiser.onnx",
    "gem_denoiser_no_imgfeat": ONNX_DIR / "gem_denoiser_no_imgfeat.onnx",
    "yolox": YOLOX_ONNX,
}

def chdir_gemx():
    os.chdir(GEMX_ROOT)
    if str(GEMX_ROOT) not in sys.path:
        sys.path.insert(0, str(GEMX_ROOT))

def load_demo():
    chdir_gemx()
    demo_py = GEMX_ROOT / "scripts" / "demo" / "demo_soma_onnx.py"
    spec = importlib.util.spec_from_file_location("gemx_demo_soma_onnx", demo_py)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo

def build_cfg(demo, video, output_root, static_cam=False):
    return demo._build_cfg(SimpleNamespace(
        video=str(video), output_root=str(output_root), static_cam=static_cam,
        verbose=False, ckpt=None, exp="gem_soma_regression"))

def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

def make_session(path, ep="cpu", threads=None, opt="all", coreml_mlprogram=True, log=print):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }[opt]
    if threads:
        so.intra_op_num_threads = threads
    providers = []
    if ep == "coreml":
        opts = {"ModelFormat": "MLProgram"} if coreml_mlprogram else {}
        providers.append(("CoreMLExecutionProvider", opts))
    providers.append("CPUExecutionProvider")
    t0 = time.time()
    sess = ort.InferenceSession(str(Path(path).resolve()), sess_options=so, providers=providers)
    log(f"[ort] {Path(path).name}: load {time.time()-t0:.1f}s EP={sess.get_providers()} rss={rss_mb():.0f}MB")
    return sess

def run_sess(sess, feed):
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, feed)
    return dict(zip(names, outs))

def read_frames(video):
    from gem.utils.video_io_utils import read_video_np
    return read_video_np(str(video))

def load_cache(preprocess_dir):
    import torch
    d = Path(preprocess_dir)
    out = {"bbx": torch.load(d / "bbx.pt", weights_only=False)}
    if (d / "vitpose.pt").exists():
        v = torch.load(d / "vitpose.pt", weights_only=False)
        out["vitpose"] = v[0] if isinstance(v, tuple) else v
    if (d / "vit_features.pt").exists():
        out["vit_features"] = torch.load(d / "vit_features.pt", weights_only=False)
    return out

def data_dict_from_cache(demo, video, preprocess_dir, no_imgfeat=False, static_cam=True):
    import torch
    from gem.utils.cam_utils import estimate_K
    from gem.utils.geo_transform import compute_cam_angvel, compute_cam_tvel
    from gem.utils.video_io_utils import get_video_lwh

    c = load_cache(preprocess_dir)
    L, W, H = get_video_lwh(str(video))
    K = estimate_K(W, H).repeat(L, 1, 1)
    T_w2c = torch.eye(4).unsqueeze(0).repeat(L, 1, 1)
    R, t = T_w2c[:, :3, :3], T_w2c[:, :3, 3]
    if no_imgfeat:
        f_imgseq, has_img = torch.zeros(L, 1024), torch.zeros(L).bool()
    else:
        f_imgseq, has_img = c["vit_features"]["pose_tokens"], torch.ones(L).bool()
    return {
        "meta": [{"vid": Path(video).stem}], "length": torch.tensor(L),
        "bbx_xys": c["bbx"]["bbx_xys"].clone(), "kp2d": c["vitpose"], "K_fullimg": K,
        "cam_angvel": compute_cam_angvel(R), "cam_tvel": compute_cam_tvel(t),
        "R_w2c": R, "T_w2c": T_w2c, "f_imgseq": f_imgseq, "noisy_pred_cam": None,
        "has_text": torch.tensor([False]),
        "mask": {"valid": torch.ones(L).bool(), "has_img_mask": has_img,
                 "has_2d_mask": torch.ones(L).bool(), "has_cam_mask": torch.ones(L).bool(),
                 "has_audio_mask": torch.zeros(L).bool(), "has_music_mask": torch.zeros(L).bool()},
    }

def soma_layer():
    chdir_gemx()
    from gem.utils.soma_utils.soma_layer import SomaLayer
    return SomaLayer(data_root="inputs/soma_assets", low_lod=True, device="cpu",
                     identity_model_type="mhr", mode="warp")

def fk_joints(soma, params) -> np.ndarray:
    import torch
    p = {k: torch.as_tensor(v).float().cpu() for k, v in params.items()
         if k in ("global_orient", "body_pose", "identity_coeffs", "scale_params", "transl")}
    with torch.no_grad():
        return soma(**p)["joints"].numpy()

def params_to_bvh(soma, params, fps):
    sys.path.insert(0, "/Users/anuraag/Developer/BAMM_API/services/gemx")
    import torch
    from soma_bvh import soma_params_to_bvh
    with torch.no_grad():
        rest = soma.get_skeleton(torch.as_tensor(params["identity_coeffs"][:1]).float(),
                                 torch.as_tensor(params["scale_params"][:1]).float())[0].numpy()
    return soma_params_to_bvh(params, rest, fps, str(GEMX_ROOT / "inputs" / "soma_assets"))

def joint_metrics(ja: np.ndarray, jb: np.ndarray, root_align=True) -> dict:
    ja, jb = ja.astype(np.float64), jb.astype(np.float64)
    d = np.linalg.norm(ja - jb, axis=-1)
    out = {"mpjpe_mm": float(d.mean() * 1000), "max_mm": float(d.max() * 1000),
           "root_mm": float(np.linalg.norm(ja[:, 0] - jb[:, 0], axis=-1).mean() * 1000)}
    if root_align:
        a, b = ja - ja[:, :1], jb - jb[:, :1]
        out["pa_root_mpjpe_mm"] = float(np.linalg.norm(a - b, axis=-1).mean() * 1000)
    return out

def rotvec_angle_deg(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.spatial.transform import Rotation as Rot
    ra = Rot.from_rotvec(a.reshape(-1, 3)); rb = Rot.from_rotvec(b.reshape(-1, 3))
    ang = np.degrees((ra.inv() * rb).magnitude())
    return {"mean_deg": float(ang.mean()), "p95_deg": float(np.percentile(ang, 95)), "max_deg": float(ang.max())}

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[saved] {path}")

def bvh_parse(text: str):
    toks = text.replace("\n", " ").replace("\t", " ").split()
    i = toks.index("HIERARCHY") + 1
    names, parents, offsets, channels, stack = [], [], [], [], []
    while toks[i] != "MOTION":
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            names.append(toks[i + 1]); parents.append(stack[-1] if stack else -1)
            offsets.append(None); channels.append([]); stack.append(len(names) - 1); i += 2
        elif t == "End":
            names.append(names[stack[-1]] + "_End"); parents.append(stack[-1])
            offsets.append(None); channels.append([]); stack.append(len(names) - 1); i += 2
        elif t == "{": i += 1
        elif t == "}": stack.pop(); i += 1
        elif t == "OFFSET":
            offsets[stack[-1]] = [float(x) for x in toks[i + 1:i + 4]]; i += 4
        elif t == "CHANNELS":
            n = int(toks[i + 1]); channels[stack[-1]] = toks[i + 2:i + 2 + n]; i += 2 + n
        else:
            raise ValueError(f"unexpected token {t}")
    i += 1
    assert toks[i] == "Frames:"; nf = int(toks[i + 1]); i += 2
    assert toks[i] == "Frame" and toks[i + 1] == "Time:"; ft = float(toks[i + 2]); i += 3
    vals = np.array([float(x) for x in toks[i:]], dtype=np.float64)
    nch = sum(len(c) for c in channels)
    frames = vals[: nf * nch].reshape(nf, nch)
    return names, parents, np.array(offsets, dtype=np.float64), channels, frames, ft

def bvh_fk(text: str):
    from scipy.spatial.transform import Rotation as Rot
    names, parents, offsets, channels, frames, ft = bvh_parse(text)
    F, J = frames.shape[0], len(names)
    pos = np.zeros((F, J, 3)); rot = [None] * J
    for f in range(F):
        c = 0
        for j in range(J):
            ch = channels[j]; vals = frames[f, c:c + len(ch)]; c += len(ch)
            p = offsets[j].copy(); R = Rot.identity()
            order, angles = "", []
            for name, v in zip(ch, vals):
                if name.endswith("position"): p["XYZ".index(name[0])] += v
                else: order += name[0]; angles.append(v)
            if order:
                R = Rot.from_euler(order.upper(), angles, degrees=True)
            if parents[j] < 0:
                pos[f, j] = p; rot[j] = R
            else:
                pos[f, j] = pos[f, parents[j]] + rot[parents[j]].apply(p); rot[j] = rot[parents[j]] * R
    return names, pos, ft

def bvh_sanity(bvh: bytes) -> list[str]:
    sys.path.insert(0, "/Users/anuraag/Developer/BAMM_API/scripts")
    from smoke_test import bvh_sanity as _s
    return _s(bvh)
