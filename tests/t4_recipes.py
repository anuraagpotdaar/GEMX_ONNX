import os, sys, time
from pathlib import Path
import onnx, onnxruntime as ort
sys.path.insert(0, str(Path(__file__).parent)); import common as C
C.chdir_gemx()
src = str(C.MODELS.get(sys.argv[1], Path(sys.argv[1]))); stem = Path(src).stem
recipes = sys.argv[2:] or ["fp16_block", "fp16_ort", "w8", "w4", "int8pc"]
def check(out):
    ort.InferenceSession(out, providers=["CPUExecutionProvider"]); return f"LOAD OK {os.path.getsize(out)/1e6:.0f} MB"
for r in recipes:
    out = str(C.QUANT / f"{stem}_{r}.onnx"); t0 = time.time()
    try:
        if r == "fp16_block":
            from onnxconverter_common import float16
            block = list(float16.DEFAULT_OP_BLOCK_LIST) + ["Cast", "Concat", "Where", "Range", "Expand", "Slice", "Gather", "ScatterND", "ConstantOfShape", "Einsum", "Pow", "Sqrt", "Div", "Softmax", "LayerNormalization", "Reshape", "Unsqueeze", "Squeeze", "Transpose"]
            onnx.save(float16.convert_float_to_float16(onnx.load(src), keep_io_types=True, disable_shape_infer=True, op_block_list=block), out)
        elif r == "fp16_ort":
            from onnxruntime.transformers.onnx_model import OnnxModel
            om = OnnxModel(onnx.load(src)); om.convert_float_to_float16(keep_io_types=True); om.save_model_to_file(out)
        elif r in ("w8", "w4"):
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
            q = MatMulNBitsQuantizer(onnx.load(src), block_size=32, is_symmetric=True, bits=int(r[1])); q.process(); q.model.save_model_to_file(out)
        elif r == "int8pc":
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(src, out, weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"])
        print(f"{r}: {check(out)} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"{r}: FAIL {type(e).__name__}: {str(e)[:220]}", flush=True)
