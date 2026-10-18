import os, sys, time, numpy as np, torch
GEMX = '/Users/anuraag/Developer/GEM-X'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(GEMX); sys.path.insert(0, GEMX)
from gem.utils.vitpose_extractor import _build_model_local
import coremltools as ct
import coremltools.optimize.coreml as cto

pose = _build_model_local("Dinov3_ViTPose_huge_metrosim_256x192", "inputs/checkpoints/vitpose/vitpose.pth").float().eval()
rope = pose.backbone.rope_embed
with torch.no_grad(): _sc = rope(H=16, W=12)
_sin, _cos = _sc[0].detach().clone(), _sc[1].detach().clone()
rope.forward = lambda *a, **k: (_sin, _cos)
from coremltools.converters.mil.frontend.torch import ops as tops
from coremltools.converters.mil import Builder as mb
_orig_cast = tops._cast
def _cast_patched(context, node, dtype, dtype_name):
    x = context[node.inputs[0]]; v = getattr(x, "val", None)
    if v is not None and hasattr(v, "shape") and v.shape != () and v.size == 1:
        context.add(mb.const(val=dtype(v.reshape(()).item()), name=node.name)); return
    return _orig_cast(context, node, dtype, dtype_name)
tops._cast = _cast_patched

with torch.no_grad(): traced = torch.jit.trace(pose, torch.zeros(1, 3, 256, 192), check_trace=False)
t0 = time.time()
ml = ct.convert(traced, inputs=[ct.TensorType(name="imgs", shape=(1, 3, 256, 192))], outputs=[ct.TensorType(name="heatmaps")],
                convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS17)
fp16 = os.path.join(ROOT, "quant", "vitpose_fp16.mlpackage"); ml.save(fp16); print("fp16 saved", fp16, round(time.time() - t0, 1), "s", flush=True)
t0 = time.time()
cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel"))
w8 = cto.linear_quantize_weights(ct.models.MLModel(fp16, compute_units=ct.ComputeUnit.CPU_ONLY), config=cfg)
out = os.path.join(ROOT, "quant", "vitpose_w8.mlpackage"); w8.save(out); print("w8 saved", out, round(time.time() - t0, 1), "s")
