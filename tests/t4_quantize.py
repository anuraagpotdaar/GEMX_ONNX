import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common as C

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("mode", choices=["fp16", "int8", "consolidate"])
ap.add_argument("--out", default=None); ap.add_argument("--keep-io", action="store_true", default=True)
a = ap.parse_args()
C.chdir_gemx()
src = C.MODELS.get(a.model, Path(a.model)); stem = Path(src).stem
out = Path(a.out) if a.out else C.QUANT / f"{stem}_{a.mode}.onnx"
out.parent.mkdir(parents=True, exist_ok=True)
import onnx
t0 = time.time()
def size(p):
    p = Path(p); return p.stat().st_size + (Path(str(p) + ".data").stat().st_size if Path(str(p) + ".data").exists() else 0)
def save(m, p):
    big = m.ByteSize() > 1_800_000_000 if hasattr(m, "ByteSize") else True
    try:
        onnx.save_model(m, str(p), save_as_external_data=big, all_tensors_to_one_file=True, location=p.name + ".data" if big else None, size_threshold=1024 if big else 1024, convert_attribute=False)
    except ValueError:
        onnx.save_model(m, str(p), save_as_external_data=True, all_tensors_to_one_file=True, location=p.name + ".data", size_threshold=1024, convert_attribute=False)
if a.mode == "consolidate":
    m = onnx.load(str(src)); save(m, out)
elif a.mode == "fp16":
    from onnxconverter_common import float16
    m = onnx.load(str(src))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)
    save(m16, out)
elif a.mode == "int8":
    from onnxruntime.quantization import quantize_dynamic, QuantType
    big = size(src) > 1_800_000_000
    quantize_dynamic(str(src), str(out), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"], use_external_data_format=big)
rep = {"src": str(src), "mode": a.mode, "out": str(out), "src_bytes": size(src), "out_bytes": size(out), "ratio": size(out) / size(src), "seconds": time.time() - t0, "rss_mb": C.rss_mb()}
C.save_json(C.REPORTS / f"t4_quant_{stem}_{a.mode}.json", rep); print(rep)
