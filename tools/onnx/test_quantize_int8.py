"""Check Q/DQ construction, per-channel axes, and smoothing on a small graph."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import ml_dtypes
import numpy as np
import onnx
from onnx import helper, numpy_helper

from fetch_models import sha256
from quantize_int8 import quantize


class QuantizationTest(unittest.TestCase):
    def test_weight_reconstruction_and_output_protection(self):
        for alpha in [None, 0.5]:
            with self.subTest(alpha=alpha), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.onnx"
                weight = np.array([[0.3, -0.7], [1.2, 0.2], [-0.4, 1.5]], dtype=ml_dtypes.bfloat16)
                node = helper.make_node("MatMul", ["x", "w"], ["y"], name="linear")
                prop = node.metadata_props.add()
                prop.key = "pkg.torch.onnx.name_scopes"
                prop.value = "['model', 'model.blocks.0.mlp.fc1', 'linear']"
                graph = helper.make_graph([node], "test", [helper.make_tensor_value_info("x", onnx.TensorProto.BFLOAT16, [1, 3])],
                                          [helper.make_tensor_value_info("y", onnx.TensorProto.BFLOAT16, [1, 2])],
                                          [numpy_helper.from_array(weight, "w")])
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
                onnx.save_model(model, source)
                source.with_suffix(".manifest.json").write_text(json.dumps(dict(
                    precision="bf16", onnx_sha256=sha256(source), checkpoint_sha256="test", external_data=[])))
                scales = root / "scales.json"
                scales.write_text("{}")
                args = argparse.Namespace(onnx=source, output=root / "int8.onnx", scales=scales,
                                          smooth_alpha=alpha, include=None)
                quantize(args, dict(checkpoint_sha256="test", fixtures=[],
                                    channels={"model.blocks.0.mlp.fc1": [100, 1, 0.1]}))
                output = onnx.load(args.output)
                onnx.checker.check_model(output)
                values = {t.name: numpy_helper.to_array(t) for t in output.graph.initializer}
                self.assertNotIn("w", values)
                self.assertEqual(values["linear_int8_weight"].dtype, np.int8)
                dq = next(n for n in output.graph.node if n.name == "linear_int8_w_dequant")
                self.assertEqual(helper.get_attribute_value(dq.attribute[0]), 1)
                scale = values["linear_int8_weight_scale"].astype(np.float32)
                restored = values["linear_int8_weight"].astype(np.float32) * scale
                smooth = np.ones(3, dtype=np.float32)
                if alpha is not None:
                    smooth = values["linear_int8_smooth"].astype(np.float32)
                    restored /= smooth[:, None]
                # Smoothing redistributes weight error. Check the quantization
                # grid bound, including clipping from rounded BF16 scales.
                transformed = weight.astype(np.float32) * smooth[:, None]
                clipping = np.maximum(np.abs(transformed).max(axis=0) - 127 * scale, 0)
                bound = (0.501 * scale + clipping)[None, :] / smooth[:, None] + 1e-6
                self.assertTrue(np.all(np.abs(restored - weight.astype(np.float32)) <= bound))
                before = args.output.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    quantize(args, dict(checkpoint_sha256="test", fixtures=[],
                                        channels={"model.blocks.0.mlp.fc1": [100, 1, 0.1]}))
                self.assertEqual(args.output.read_bytes(), before)
                manifest_path = source.with_suffix('.manifest.json')
                manifest = json.loads(manifest_path.read_text())
                manifest['fp32_residuals'] = True
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, 'matching precision policy'):
                    quantize(args, dict(checkpoint_sha256='test', fixtures=[],
                                        channels={'model.blocks.0.mlp.fc1':[100,1,.1]}))
                self.assertEqual(args.output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
