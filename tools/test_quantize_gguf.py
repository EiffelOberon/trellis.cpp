"""Regression: 2-D normalization parameters are not quantized matmul weights."""
from pathlib import Path
import tempfile
import unittest

import gguf
import numpy as np

from quantize_gguf import quantize_file


class QuantizeTest(unittest.TestCase):
    def test_preserves_elementwise_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            src, dst = (Path(directory) / name for name in ('source.gguf', 'q8.gguf'))
            writer = gguf.GGUFWriter(src, 'trellis')
            tensors = {
                'blocks.0.self_attn.to_qkv.weight': np.ones((384, 128), dtype=np.float16),
                'blocks.0.self_attn.q_rms_norm.gamma': np.ones((12, 128), dtype=np.float16),
                'blocks.0.modulation': np.ones((6, 128), dtype=np.float16),
                'input_layer.weight': np.ones((128, 32), dtype=np.float16),
            }
            for name, value in tensors.items():
                writer.add_tensor(name, value)
            writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
            quantize_file(src, dst, 'Q8_0')
            actual = {t.name: (t.tensor_type, t.data.copy()) for t in gguf.GGUFReader(dst).tensors}
            self.assertEqual(actual['blocks.0.self_attn.to_qkv.weight'][0], gguf.GGMLQuantizationType.Q8_0)
            for name, value in tensors.items():
                if name == 'blocks.0.self_attn.to_qkv.weight':
                    continue
                self.assertEqual(actual[name][0], gguf.GGMLQuantizationType.F16)
                np.testing.assert_array_equal(actual[name][1], value)


if __name__ == '__main__':
    unittest.main()
