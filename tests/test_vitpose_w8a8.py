"""Regression checks for calibration accumulation and the SwiGLU guard."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types
import numpy as np
from tools.quantize_vitpose_w8a8 import update_range, load_ranges, main_block, apply_swiglu_fix


class CalibrationTests(unittest.TestCase):
    def test_extrema_survive_later_samples(self):
        ranges = {}
        update_range(ranges, 'x', np.array([-7., 2.]))
        update_range(ranges, 'x', np.array([-1., 9.]))
        self.assertEqual(ranges['x'], {'rmin': -7., 'rmax': 9.})

    def test_nonfinite_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            update_range({}, 'x', np.array([1., np.nan]))

    def test_stale_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'cache.json'
            p.write_text(json.dumps({'identity': {'source': 'old'}, 'ranges': {}}))
            with self.assertRaises(ValueError):
                load_ranges(p, {'source': 'new'})

    @unittest.skipUnless(sys.platform == 'darwin', 'Core ML prediction requires macOS')
    def test_swiglu_guard_preserves_cpu_saturation(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4,), dtype=types.fp16),
                                 mb.TensorSpec(shape=(4,), dtype=types.fp16)],
                    opset_version=ct.target.iOS17)
        def program(x, gate):
            product = mb.mul(x=mb.silu(x=x), y=gate)
            q = mb.quantize(input=product, scale=np.float16(.125), zero_point=np.int8(0), output_dtype='int8')
            return mb.dequantize(input=q, scale=np.float16(.125), zero_point=np.int8(0))
        original = ct.convert(program, minimum_deployment_target=ct.target.iOS17, skip_model_load=True)
        quantizer = next(o for o in main_block(original.get_spec()).operations if o.type == 'quantize')
        name = quantizer.inputs['input'].arguments[0].name
        fixed, audit = apply_swiglu_fix(original, {name: {'rmin': -15.875, 'rmax': 15.875}}, 1, 1)
        self.assertEqual(len(audit['clamps']), 1)
        inputs = {'x': np.array([50, 50, 1, -1], np.float16),
                  'gate': np.array([-50, 50, .25, -.5], np.float16)}
        outputs = []
        for model in [original, fixed]:
            loaded = ct.models.MLModel(model.get_spec(), weights_dir=model.weights_dir, compute_units=ct.ComputeUnit.CPU_ONLY)
            outputs.append(next(iter(loaded.predict(inputs).values())))
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertEqual(float(outputs[1][0]), -16.)
        self.assertEqual(float(outputs[1][1]), 15.875)


if __name__ == '__main__':
    unittest.main()
