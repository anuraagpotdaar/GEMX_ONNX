import argparse, sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import common as C
sys.path.insert(0, str(Path(__file__).parent.parent / "phone")); import gemx_phone as P
ap = argparse.ArgumentParser(); ap.add_argument("--clip", default="test3_demo_6fps"); ap.add_argument("--e2e", action="store_true"); ap.add_argument("--step", type=int, default=3); ap.add_argument("--threads", type=int, default=None)
a = ap.parse_args(); clip = a.clip
demo = C.load_demo(); demo._ensure_pipeline_deps()
import torch, hydra
video = C.GEMX_ROOT / "sample_inputs" / f"{clip}.mp4"; cache = C.RUNS / "bench" / clip / clip / "preprocess"
out_dir = C.RUNS / "phone" / clip; out_dir.mkdir(parents=True, exist_ok=True); rep = {"clip": clip}
pp = P.PhonePipeline(threads=a.threads); t_load = pp.load(); rep["t_load_s"] = t_load
data = C.data_dict_from_cache(demo, video, cache, no_imgfeat=True); kp_ref = data["kp2d"].numpy(); bb_ref = data["bbx_xys"].numpy(); N = len(kp_ref)
cfg = C.build_cfg(demo, video, C.RUNS / "_cfg_tmp", static_cam=True)
model = hydra.utils.instantiate(cfg.model, _recursive_=False); model.load_pretrained_model("inputs/pretrained/gem_soma.ckpt"); model = model.eval()
ende = model.endecoder; ende.build_obs_indices_dict() if ende.obs_indices_dict is None else None
soma = C.soma_layer(); W, H = int(data["K_fullimg"][0, 0, 2] * 2), int(data["K_fullimg"][0, 1, 2] * 2)
def log(k, v): rep[k] = v; print("[t8]", k, json.dumps(v, default=float), flush=True)

r = pp.run(None, fps=30, kp2d_full=kp_ref, bbx_xys_full=bb_ref, window=10**9, step=1, size=(W, H))
with torch.no_grad():
    dd = ende.decode(torch.from_numpy(r["pred_x"])[None]); from gem.pipeline.gem_pipeline import get_body_params_w_Rt_v2
    g = get_body_params_w_Rt_v2(global_orient_gv=dd["global_orient_gv"], local_transl_vel=dd["local_transl_vel"], global_orient_c=dd["global_orient"], cam_angvel=data["cam_angvel"][None])
    from gem.utils.cam_utils import compute_transl_full_cam; tc = compute_transl_full_cam(torch.from_numpy(r["pred_cam"])[None], data["bbx_xys"][None], data["K_fullimg"][None])[0].numpy()
ang = lambda x, y: C.rotvec_angle_deg(np.asarray(x), np.asarray(y))["max_deg"]

d = P.decode_x(r["pred_x"], pp.tab["x_mean"], pp.tab["x_std"]); go, tr = P.global_frame_static(d["global_orient_gv"], d["local_transl_vel"])
log("A_decode_vs_pytorch", {"body_pose_max_deg": ang(d["body_pose"].reshape(-1, 3), dd["body_pose"][0].reshape(-1, 3).numpy()), "global_orient_max_deg": ang(go, g["global_orient"][0].numpy()),
    "transl_max_mm": float(np.abs(tr - g["transl"][0].numpy()).max() * 1000), "identity_max": float(np.abs(d["identity_coeffs"] - dd["identity_coeffs"][0].numpy()).max()),
    "scale_max": float(np.abs(d["scale_params"][:, 1:] - dd["scale_params"][0, :, 1:].numpy()).max()), "incam_transl_max_mm": float(np.abs(r["incam"]["transl"] - tc).max() * 1000)})

p_ref = torch.load(C.RUNS / "visual" / clip / "pytorch_postproc_hpe_results.pt", weights_only=False)["body_params_global"]; pr = {k: v.numpy() for k, v in p_ref.items()}
with torch.no_grad(): rest_soma = soma.get_skeleton(p_ref["identity_coeffs"][:1].float(), p_ref["scale_params"][:1].float())[0].numpy()
rest_lut = P.rest_skeleton(pr["identity_coeffs"][0], pr["scale_params"][0], pp.lut)
j_soma = C.fk_joints(soma, pr); j_fk_exact = P.fk(rest_soma, pr["global_orient"], pr["body_pose"], pr["transl"], pp.parents); j_fk_lut = P.fk(rest_lut, pr["global_orient"], pr["body_pose"], pr["transl"], pp.parents)
log("B_skeleton_fk", {"rest_lut_vs_soma_max_mm": float(np.linalg.norm(rest_lut - rest_soma, axis=-1).max() * 1000), "fk_exact_rest_vs_soma_max_mm": float(np.linalg.norm(j_fk_exact - j_soma, axis=-1).max() * 1000),
    "fk_lut_rest_vs_soma_max_mm": float(np.linalg.norm(j_fk_lut - j_soma, axis=-1).max() * 1000)})

bvh = P.to_bvh(pr["global_orient"], pr["body_pose"], pr["transl"], rest_soma, pp.names, pp.parents, 30); names, pos, _ = C.bvh_fk(bvh)
keep = [i for i, n in enumerate(names) if not n.endswith("_End")]; log("B_bvh_roundtrip_max_mm", float(np.linalg.norm(pos[:, keep] - j_soma, axis=-1).max() * 1000)); rep["bvh_sanity_as_is_params"] = C.bvh_sanity(bvh.encode())

frames = C.read_frames(video); ii = np.arange(0, min(N, 8)); my = P.vitpose_run(pp.vp, frames, ii, bb_ref[ii])
imgs, _ = demo.vitpose_preprocess(frames[ii], torch.from_numpy(bb_ref[ii])); hm = pp.vp.run(["heatmaps"], {"imgs": imgs[:, :, :, 32:224].numpy()})[0]
ref = demo.vitpose_postprocess(hm, torch.from_numpy(bb_ref[ii])).numpy()
log("C_vitpose_vs_demo_helpers", {"max_px": float(np.abs(my[..., :2] - ref[..., :2]).max()), "conf_max_diff": float(np.abs(my[..., 2] - ref[..., 2]).max()),
    "int8_noflip_vs_mps_flip_mean_px": float(np.linalg.norm(my[..., :2] - kp_ref[ii, :, :2], axis=-1).mean())})

idx = P.sample_indices(N, a.step); t0 = time.time(); xy = P.smooth_xyxy(P.primary_boxes(frames, idx, pp.det, W, H)); t_det = time.time() - t0
bb_s = P.xyxy_to_xys(xy, W, H); bb_i = P.interp_frames(bb_s, idx, N)
log("D_detector_vs_bytetrack", {"center_mean_px": float(np.linalg.norm(bb_i[:, :2] - bb_ref[:, :2], axis=-1).mean()), "size_mean_rel": float((np.abs(bb_i[:, 2] - bb_ref[:, 2]) / bb_ref[:, 2]).mean()), "t_detect_s": t_det, "n_detect": int(len(idx))})

if a.e2e:
    t0 = time.time(); r = pp.run(video, step=a.step, max_side=1080); t_total = time.time() - t0; (out_dir / "phone.bvh").write_text(r["bvh"]); np.savez(out_dir / "phone_out.npz", **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
    j_asis = C.fk_joints(soma, pr); m_asis = C.joint_metrics(r["joints"], j_asis)
    e_ref = torch.load(C.RUNS / "visual" / clip / "edge_vitpose_int8_noimg_fp16_hpe_results.pt", weights_only=False)["body_params_global"]; j_edge = C.fk_joints(soma, {k: v.numpy() for k, v in e_ref.items()})
    m_edge = C.joint_metrics(r["joints"], j_edge)

    d2 = dict(data); d2["kp2d"] = torch.from_numpy(r["kp2d"]); d2["bbx_xys"] = torch.from_numpy(r["bbx_xys"]); d2["K_fullimg"] = torch.from_numpy(P.estimate_K(*r["size"]))[None].repeat(N, 1, 1)
    with torch.no_grad(): pt = model.predict(d2, static_cam=True, postproc=False)
    j_pt = C.fk_joints(soma, {k: v.numpy() for k, v in pt["body_params_global"].items()}); m_pt = C.joint_metrics(r["joints"], j_pt)
    log("E_e2e", {"frames": r["frames"], "sampled": r["sampled"], "size": r["size"], "t_total_s": t_total, "timings": r["timings"], "bvh_sanity": C.bvh_sanity(r["bvh"].encode()),
        "pa_mm_vs_as_is": m_asis["pa_root_mpjpe_mm"], "mpjpe_mm_vs_as_is": m_asis["mpjpe_mm"], "pa_mm_vs_edge_allframes_flip": m_edge["pa_root_mpjpe_mm"],
        "pa_mm_vs_pytorch_same_inputs": m_pt["pa_root_mpjpe_mm"], "mpjpe_mm_vs_pytorch_same_inputs_ungrounded": m_pt["mpjpe_mm"], "height_m": float(r["rest"][:, 1].max() - r["rest"][:, 1].min())})
if a.e2e:

    kp_sub = P.interp_frames(kp_ref[idx], idx, N); kp_s = P.vitpose_run(pp.vp, frames, idx, bb_ref[idx]); kp_phone_bt = P.interp_frames(kp_s, idx, N)
    variants = {"cached_kp2d_all+bytetrack_bbx": (kp_ref, bb_ref), "cached_kp2d_subsampled+bytetrack_bbx": (kp_sub, bb_ref), "phone_kp2d_sampled+bytetrack_bbx": (kp_phone_bt, bb_ref), "phone_kp2d+phone_bbx(fullres)": (kp_phone_bt, bb_i)}
    att = {}
    for name, (kp, bb) in variants.items():
        rr = pp.run(None, fps=30, kp2d_full=kp, bbx_xys_full=bb, size=(W, H))
        att[name] = {"pa_vs_as_is": C.joint_metrics(rr["joints"], j_asis)["pa_root_mpjpe_mm"], "pa_vs_edge": C.joint_metrics(rr["joints"], j_edge)["pa_root_mpjpe_mm"]}
    log("E_attribution", att)
C.save_json(out_dir / "validate.json", rep)
