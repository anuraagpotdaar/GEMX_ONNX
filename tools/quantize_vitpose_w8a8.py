"""Calibrate W8 ViTPose and export only the SwiGLU-fixed W8A8 model.

Input crops must use the app's normalization/layout: float32 NCHW, 256 x 192.
The existing INT8 weights are preserved. Only linear/conv activation regions
are quantized; attention, normalization and other regions remain floating point.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.optimize.coreml import _post_training_quantization as ptq
from coremltools.converters.mil.mil import Builder as mb, types
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main_block(spec):
    return next(iter(spec.mlProgram.functions['main'].block_specializations.values()))


def update_range(ranges, name, value):
    lo, hi = float(np.min(value)), float(np.max(value))
    if not np.isfinite([lo, hi]).all():
        raise ValueError(f'Non-finite calibration tensor: {name}')
    old = ranges.get(name)
    ranges[name] = {'rmin': min(lo, old['rmin']) if old else lo,
                    'rmax': max(hi, old['rmax']) if old else hi}


def cache_identity(source, crops):
    return {'coremltools': ct.__version__, 'flip': True,
            'source_graph_sha256': sha256(source / 'Data/com.apple.CoreML/model.mlmodel'),
            'source_weights_sha256': sha256(source / 'Data/com.apple.CoreML/weights/weight.bin'),
            'crops_sha256': sha256(crops)}


def load_ranges(path, identity):
    if not path.exists():
        return {}
    cached = json.loads(path.read_text())
    if cached.get('identity') != identity:
        raise ValueError('Calibration cache belongs to different weights, graph, crops or tools. Use a new --work-dir.')
    return cached['ranges']


def calibrate(model, data, cache, identity, group_size):
    spec = model.get_spec()
    ops = main_block(spec).operations
    tensor_types = {v.name: v.type for op in ops for v in op.outputs}
    names = set()
    for op in ops:
        if op.type in ['linear', 'conv']:
            names.update(a.name for a in op.inputs['x'].arguments if a.name)
            names.update(v.name for v in op.outputs)
    ordered = list(dict.fromkeys(v.name for op in ops for v in op.outputs if v.name in names))
    ranges = load_ranges(cache, identity)
    for sample in data:
        update_range(ranges, 'imgs', sample['imgs'])
    missing = [name for name in ordered if name not in ranges]
    print(f'Calibration: {len(data)} inputs, {len(missing)} uncached boundaries', flush=True)
    # Compile once per group, then union extrema across every input. This avoids
    # the coremltools 9.0 collector resetting its accumulator between samples.
    for start in range(0, len(missing), group_size):
        group = missing[start:start + group_size]
        tmp = type(spec)(); tmp.CopyFrom(spec)
        block = main_block(tmp)
        del tmp.description.output[:]; del block.outputs[:]
        for name in group:
            block.outputs.append(name)
            output = tmp.description.output.add(); output.name = name
            if tensor_types[name].tensorType.dataType != ct.proto.MIL_pb2.FLOAT16:
                raise ValueError(f'Expected FP16 activation: {name}')
            output.type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16
            output.type.multiArrayType.shape.extend(d.constant.size for d in tensor_types[name].tensorType.dimensions)
        probe = ct.models.MLModel(tmp, weights_dir=model.weights_dir, compute_units=ct.ComputeUnit.CPU_ONLY)
        for sample in data:
            result = probe.predict(sample)
            for name in group:
                update_range(ranges, name, result[name])
        cache.write_text(json.dumps({'identity': identity, 'ranges': ranges}, indent=2))
        print(f'Calibrated {min(start + group_size, len(missing))}/{len(missing)} boundaries', flush=True)
        del probe, tmp, result
        gc.collect()
    if not names.issubset(ranges):
        raise ValueError('Incomplete calibration')
    return ranges


def copy_op(op):
    result = type(op)(); result.CopyFrom(op)
    return result


def rename(op, name):
    op.attributes['name'].immediateValue.tensor.strings.values[0] = name
    op.outputs[0].name = name


def fp16_scalar(op):
    value = op.attributes['val']
    if value.type.tensorType.dataType != ct.proto.MIL_pb2.FLOAT16:
        raise ValueError('Expected an FP16 quantization scale')
    return np.frombuffer(value.immediateValue.tensor.bytes.values, np.float16).item()


def apply_swiglu_fix(model, ranges, expected_clamps=32, expected_quantizers=291):
    """Make INT8 saturation explicit, avoiding the observed ANE mul/QDQ issue."""
    @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp16)], opset_version=ct.target.iOS17)
    def clip_program(x):
        return mb.clip(x=x, alpha=np.float16(-1), beta=np.float16(1))
    template = ct.convert(clip_program, minimum_deployment_target=ct.target.iOS17, skip_model_load=True)
    clip_template = next(op for op in main_block(template.get_spec()).operations if op.type == 'clip')
    spec = model.get_spec(); block = main_block(spec)
    producers = {v.name: op for op in block.operations for v in op.outputs}
    new_ops, changes, scale_audit = [], [], []
    for old in block.operations:
        op = copy_op(old)
        if op.type == 'quantize':
            name = op.inputs['input'].arguments[0].name
            scale_op = producers[op.inputs['scale'].arguments[0].name]
            scale = fp16_scalar(scale_op)
            recorded = ranges[name]
            expected = float(np.float16(max(abs(recorded['rmin']), abs(recorded['rmax'])) / 127))
            if not np.isfinite(scale) or scale <= 0 or scale != expected:
                raise ValueError(f'Unexpected calibration scale at {name}: {scale} != {expected}')
            scale_audit.append({'input': name, 'scale': scale})
            parent = producers.get(name)
            if parent is not None and parent.type == 'mul':
                if not any(producers[a.name].type == 'silu'
                           for v in parent.inputs.values() for a in v.arguments if a.name in producers):
                    raise ValueError(f'Unexpected non-SwiGLU multiply at {name}')
                clip_name = name + '_pre_quant_clip'
                for suffix, value in [('min', -128 * scale), ('max', 127 * scale)]:
                    constant = copy_op(scale_op); rename(constant, clip_name + '_' + suffix)
                    constant.attributes['val'].immediateValue.tensor.bytes.values = np.float16(value).tobytes()
                    new_ops.append(constant)
                clip = copy_op(clip_template); rename(clip, clip_name)
                clip.inputs['x'].arguments[0].name = name
                clip.inputs['alpha'].arguments[0].name = clip_name + '_min'
                clip.inputs['beta'].arguments[0].name = clip_name + '_max'
                clip.outputs[0].type.CopyFrom(parent.outputs[0].type)
                new_ops.append(clip)
                op.inputs['input'].arguments[0].name = clip_name
                changes.append({'input': name, 'scale': scale, 'min': -128 * scale, 'max': 127 * scale})
        new_ops.append(op)
    if len(changes) != expected_clamps or len(scale_audit) != expected_quantizers:
        raise ValueError(f'Unexpected graph coverage: {len(changes)} clamps, {len(scale_audit)} quantizers')
    del block.operations[:]; block.operations.extend(new_ops)
    fixed = ct.models.MLModel(spec, weights_dir=model.weights_dir, skip_model_load=True)
    return fixed, {'clamps': changes, 'scales': scale_audit}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT / 'quant/vitpose_w8.mlpackage')
    ap.add_argument('--crops', type=Path, required=True, help='Normalized float32 crops.f32, shape N x 3 x 256 x 192')
    ap.add_argument('--output', type=Path, default=ROOT / 'quant/vitpose_w8a8_clipped.mlpackage')
    ap.add_argument('--work-dir', type=Path, default=ROOT / 'runs/vitpose_w8a8')
    ap.add_argument('--group-size', type=int, default=32)
    args = ap.parse_args()
    if ct.__version__ != '9.0':
        raise RuntimeError('This calibration hook is validated with coremltools 9.0 only.')
    if args.output.exists():
        ap.error('Output already exists; choose a new --output to preserve the validated model.')
    if args.group_size < 1:
        ap.error('--group-size must be positive')
    crops = np.fromfile(args.crops, np.float32).reshape(-1, 3, 256, 192)
    if len(crops) == 0 or not np.isfinite(crops).all():
        ap.error('Crops must contain finite, nonempty normalized input tensors')
    data = [{'imgs': np.ascontiguousarray((crop[..., ::-1] if flip else crop)[None], dtype=np.float16)}
            for crop in crops for flip in [False, True]]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    identity = cache_identity(args.source, args.crops)
    model = ct.models.MLModel(str(args.source), skip_model_load=True)
    ranges = calibrate(model, data, args.work_dir / 'calibration.json', identity, args.group_size)
    config = cto.OptimizationConfig(global_config=None, op_type_configs={
        op: cto.OpLinearQuantizerConfig(mode='linear_symmetric', dtype='int8') for op in ['linear', 'conv']})
    original = ptq._get_activation_calibration_stats
    ptq._get_activation_calibration_stats = lambda *a, **k: ranges
    try:
        quantized = cto.linear_quantize_activations(model, config, data)
    finally:
        ptq._get_activation_calibration_stats = original
    fixed, audit = apply_swiglu_fix(quantized, ranges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixed.save(str(args.output))
    output_weights = sha256(args.output / 'Data/com.apple.CoreML/weights/weight.bin')
    if output_weights != identity['source_weights_sha256']:
        raise RuntimeError('Unexpected change to the W8 weight bytes')
    audit.update({'identity': identity, 'samples_including_flip': len(data),
        'output': str(args.output.resolve()), 'weight_bytes_unchanged': True,
        'op_counts': dict(Counter(op.type for op in main_block(fixed.get_spec()).operations))})
    (args.work_dir / 'export_audit.json').write_text(json.dumps(audit, indent=2))
    print(f'Saved corrected W8A8: {args.output} (32 SwiGLU clamps)', flush=True)


if __name__ == '__main__':
    main()
