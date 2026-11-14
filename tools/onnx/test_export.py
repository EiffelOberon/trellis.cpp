"""CPU tests for the real-RoPE adaptation and fixed native tensor ABI."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
import onnxruntime as ort
import torch

from export_ss_flow import load_reference, real_rope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    load_reference(args.reference.resolve())
    from trellis2.modules.attention.rope import RotaryPositionEmbedder
    torch.manual_seed(42)
    coords = torch.stack(torch.meshgrid(*[torch.arange(3)] * 3, indexing="ij"), -1).reshape(-1, 3)
    phases = RotaryPositionEmbedder(128, 3)(coords)
    x = torch.randn(1, 27, 2, 128)
    expected = RotaryPositionEmbedder.apply_rotary_embedding(x, phases)
    actual = real_rope(x, torch.view_as_real(phases))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("phases", torch.view_as_real(phases))

        def forward(self, value):
            return real_rope(value, self.phases)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rope.onnx"
        torch.onnx.export(Probe().eval(), (x,), path, dynamo=True, opset_version=20, verbose=False,
                          input_names=["value"], output_names=["result"])
        result = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"value": x.numpy()})[0]
        np.testing.assert_allclose(result, expected.numpy(), rtol=1e-5, atol=1e-6)
    print("PASS: real RoPE agrees with upstream complex RoPE and ONNX Runtime")


if __name__ == "__main__":
    main()
