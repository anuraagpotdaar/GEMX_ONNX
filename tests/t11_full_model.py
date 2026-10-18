import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, '/Users/anuraag/Developer/GEMX_ONNX/tests'); import common as C
video = Path(sys.argv[1]).resolve(); stem = video.stem
demo = C.load_demo(); demo._ensure_pipeline_deps()
import torch, hydra, cv2
from gem.utils.cam_utils import estimate_K
fps = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) + 0.5) or 30
out_root = C.RUNS / "bench" / stem; cache = out_root / stem / "preprocess"
cfg = C.build_cfg(demo, video, out_root, static_cam=True)
data = C.data_dict_from_cache(demo, video, cache, no_imgfeat=False)
print({k: tuple(v.shape) for k, v in data.items() if isinstance(v, torch.Tensor)}, flush=True)
L = data["bbx_xys"].shape[0]
if data["K_fullimg"].shape[0] != L:
    W, H = [int(cv2.VideoCapture(str(video)).get(p)) for p in (3, 4)]
    data["K_fullimg"] = estimate_K(W, H).repeat(L, 1, 1); data["length"] = torch.tensor(L)
    for k in ("R_w2c", "T_w2c"):
        data[k] = torch.eye(4 if k == "T_w2c" else 3).unsqueeze(0).repeat(L, 1, 1) if data[k].shape[0] != L else data[k]
    from gem.utils.geo_transform import compute_cam_angvel, compute_cam_tvel
    data["cam_angvel"] = compute_cam_angvel(data["R_w2c"]); data["cam_tvel"] = compute_cam_tvel(data["T_w2c"][:, :3, 3])
    data["mask"] = {"valid": torch.ones(L).bool(), "has_img_mask": torch.ones(L).bool(), "has_2d_mask": torch.ones(L).bool(),
                    "has_cam_mask": torch.ones(L).bool(), "has_audio_mask": torch.zeros(L).bool(), "has_music_mask": torch.zeros(L).bool()}
    print("patched K/masks to L =", L, "(video W,H =", W, H, ")", flush=True)
model = hydra.utils.instantiate(cfg.model, _recursive_=False)
model.load_pretrained_model(cfg.ckpt_path or "inputs/pretrained/gem_soma.ckpt"); model = model.eval().to(demo.DEVICE)
t0 = time.time()
import os; PP = os.environ.get("POSTPROC", "1") == "1"
with torch.no_grad(): pred = model.predict(data, static_cam=True, postproc=PP)
print("denoise s", round(time.time() - t0, 1), flush=True)
glob = {k: v.detach().float().cpu().numpy() for k, v in pred["body_params_global"].items() if isinstance(v, torch.Tensor)}
soma = C.soma_layer(); J = C.fk_joints(soma, glob)
out = C.RUNS / "phone" / stem; out.mkdir(parents=True, exist_ok=True)
np.savez(out / ("full_out.npz" if PP else "full_raw_out.npz"), joints=J, kp2d=data["kp2d"].numpy(), bbx_xys=data["bbx_xys"].numpy(), **glob)
(out / ("full.bvh" if PP else "full_raw.bvh")).write_text(C.params_to_bvh(soma, glob, fps))
print("saved", J.shape, flush=True)
