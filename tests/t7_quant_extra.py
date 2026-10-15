import sys, time, os, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import common as C
C.chdir_gemx(); import onnx, onnxruntime as ort, torch
Q = C.QUANT; rep = {}
def size(p): p = Path(p); return (p.stat().st_size + (Path(str(p) + ".data").stat().st_size if Path(str(p) + ".data").exists() else 0)) / 1e9
sam_int8 = Q / "sam3db_backbone_int8.onnx"
if not sam_int8.exists():
    from onnxruntime.quantization import quantize_dynamic, QuantType
    t0 = time.time(); quantize_dynamic(str(C.MODELS["sam3db_backbone"]), str(sam_int8), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"], use_external_data_format=True)
    print("[t7b] sam3db int8 built", round(time.time() - t0), "s", round(size(sam_int8), 2), "GB", flush=True)
vp_w4 = Q / "vitpose_w4.onnx"
if not vp_w4.exists():
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    t0 = time.time(); m = onnx.load(str(C.MODELS["vitpose"])); q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True, bits=4); q.process()
    onnx.save_model(q.model.model, str(vp_w4), save_as_external_data=True, all_tensors_to_one_file=True, location=vp_w4.name + ".data", size_threshold=1024)
    del m, q; print("[t7b] vitpose w4 built", round(time.time() - t0), "s", round(size(vp_w4), 2), "GB", flush=True)
rep["sizes_gb"] = {"sam3db_int8": size(sam_int8), "vitpose_w4": size(vp_w4), "vitpose_int8": size(Q / "vitpose_int8.onnx")}
demo = C.load_demo(); demo._ensure_pipeline_deps()
video = str(C.GEMX_ROOT / "sample_inputs/test3_demo_6fps.mp4"); cache = C.RUNS / "bench/test3_demo_6fps/test3_demo_6fps/preprocess"
frames = C.read_frames(video); bbx = C.load_cache(cache)["bbx"]["bbx_xys"]; kp_ref = C.load_cache(cache)["vitpose"].numpy(); N = len(frames)

def vp_run(path, thr):
    sess = C.make_session(path, "cpu", threads=thr, log=lambda *s: None)
    class Rn:
        input_names = [i.name for i in sess.get_inputs()]
        def __call__(self, **kw):
            feed = {k: (x.cpu().numpy() if isinstance(x, torch.Tensor) else x) for k, x in kw.items()}; return {k: torch.from_numpy(x) for k, x in C.run_sess(sess, feed).items()}
    imgs, bb = demo.vitpose_preprocess(frames, bbx); res = []; t0 = time.time()
    for j in range(0, N, 8): hm = Rn()(imgs=imgs[j:j+8, :, :, 32:224])["heatmaps"].numpy(); res.append(demo.vitpose_postprocess(hm, bb[j:j+8]))
    kp = torch.cat(res).numpy(); t = time.time() - t0; del sess
    return kp, t
for name, path in (("vitpose_int8", Q / "vitpose_int8.onnx"), ("vitpose_w4", vp_w4)):
    kp, t = vp_run(path, 2); err = np.linalg.norm(kp[..., :2] - kp_ref[..., :2], axis=-1)
    rep[name] = {"ms_per_frame_2thr_noflip": t / N * 1000, "px_mean_vs_mps": float(err.mean()), "px_p95": float(np.percentile(err, 95)), "px_max": float(err.max()), "rss_mb": C.rss_mb()}
    np.save(C.RUNS / "fpsres" / f"{name}_kp2d_test3_demo_6fps.npy", kp); print("[t7b]", name, rep[name], flush=True)
del frames

from gem.utils.sam3db_extractor import SAM3DBExtractor
ext = SAM3DBExtractor(device=demo.DEVICE); sess = C.make_session(sam_int8, "cpu", log=lambda *s: None)
class R:
    input_names = [i.name for i in sess.get_inputs()]
    def __call__(self, **kw):
        feed = {k: np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32) for k, v in kw.items()}; return {k: torch.from_numpy(v) for k, v in C.run_sess(sess, feed).items()}
mdl = ext.estimator.model; mdl.backbone = demo._OnnxBackbone(R(), mdl.image_mean.clone(), mdl.image_std.clone()).to(demo.DEVICE)
t0 = time.time(); res = ext.extract_video_features(video, bbx); t_sam = time.time() - t0; tok = res["pose_tokens"]
tok_ref = C.load_cache(cache)["vit_features"]["pose_tokens"]
cos = torch.nn.functional.cosine_similarity(tok, tok_ref, dim=-1)
rep["sam3db_int8"] = {"s_per_frame_cpu_alltreads": t_sam / N, "token_cos_mean": float(cos.mean()), "token_cos_min": float(cos.min()), "rss_mb": C.rss_mb()}
del sess, ext
import hydra
data = C.data_dict_from_cache(demo, video, cache, no_imgfeat=False); cfg = C.build_cfg(demo, video, C.RUNS / "_cfg_tmp", static_cam=True)
model = hydra.utils.instantiate(cfg.model, _recursive_=False); model.load_pretrained_model("inputs/pretrained/gem_soma.ckpt"); model = model.eval().to(demo.DEVICE)
soma = C.soma_layer()
with torch.no_grad():
    ref = model.predict(data, static_cam=True, postproc=False); j_ref = C.fk_joints(soma, {k: v.cpu().numpy() for k, v in ref["body_params_global"].items()})
    d2 = dict(data); d2["f_imgseq"] = tok.float(); p = model.predict(d2, static_cam=True, postproc=False); j = C.fk_joints(soma, {k: v.cpu().numpy() for k, v in p["body_params_global"].items()})
    rep["sam3db_int8"]["pa_mm_vs_fp32_tokens"] = C.joint_metrics(j, j_ref)["pa_root_mpjpe_mm"]; rep["sam3db_int8"]["mpjpe_mm_vs_fp32_tokens"] = C.joint_metrics(j, j_ref)["mpjpe_mm"]

    kp_i8 = torch.from_numpy(np.load(C.RUNS / "fpsres" / "vitpose_int8_kp2d_test3_demo_6fps.npy")).float(); d3 = dict(d2); d3["kp2d"] = kp_i8
    p = model.predict(d3, static_cam=True, postproc=False); j = C.fk_joints(soma, {k: v.cpu().numpy() for k, v in p["body_params_global"].items()})
    rep["pro_tier_vitpose_int8_sam3db_int8"] = {"pa_mm_vs_ref": C.joint_metrics(j, j_ref)["pa_root_mpjpe_mm"], "mpjpe_mm_vs_ref": C.joint_metrics(j, j_ref)["mpjpe_mm"]}
    kp_w4 = torch.from_numpy(np.load(C.RUNS / "fpsres" / "vitpose_w4_kp2d_test3_demo_6fps.npy")).float(); d4 = dict(data); d4["kp2d"] = kp_w4; d4["f_imgseq"] = torch.zeros_like(data["f_imgseq"]); d4["mask"] = dict(data["mask"]); d4["mask"]["has_img_mask"] = torch.zeros(N).bool()
    p = model.predict(d4, static_cam=True, postproc=False); j = C.fk_joints(soma, {k: v.cpu().numpy() for k, v in p["body_params_global"].items()})
    d5 = dict(d4); d5["kp2d"] = kp_i8; p5 = model.predict(d5, static_cam=True, postproc=False); j5 = C.fk_joints(soma, {k: v.cpu().numpy() for k, v in p5["body_params_global"].items()})
    rep["low_tier_vitpose_w4_noimg"] = {"pa_mm_vs_ref": C.joint_metrics(j, j_ref)["pa_root_mpjpe_mm"], "pa_mm_vs_vitpose_int8_noimg": C.joint_metrics(j, j5)["pa_root_mpjpe_mm"]}
rep["peak_rss_mb"] = C.rss_mb(); C.save_json(C.REPORTS / "t7_quant_extra.json", rep); print("[t7b] DONE", json.dumps(rep, default=float))
