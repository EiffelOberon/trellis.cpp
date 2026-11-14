"""Export Microsoft's dense SS flow with fixed B=1, L=4096, Lc=1029.

The native ABI is token-major float32: latent[1,4096,8], timestep[1]
(already multiplied by 1000), cond[1,1029,1024] -> velocity[1,4096,8].
Python is an offline dependency only. No sparse operator or custom plugin is used.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch
from safetensors.torch import load_file
from safetensors import safe_open

from fetch_models import sha256


def load_reference(source):
    """Import only the dense model, without installing the sparse/render stack.

    Sparse classes are referenced by utils' dtype conversion type tuple, but are
    never instantiated by this model. Sentinels fail if that assumption changes.
    All executed reference model/attention/norm/sampler code remains upstream.
    """
    os.environ["ATTN_BACKEND"] = "sdpa"
    for name, directory in [("trellis2", source / "trellis2"),
                            ("trellis2.models", source / "trellis2/models")]:
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        sys.modules[name] = module
    sparse = types.ModuleType("trellis2.modules.sparse")

    class UnavailableSparse(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("A sparse operator was requested by the dense exporter")

    for name in ["SparseConv3d", "SparseInverseConv3d", "SparseLinear"]:
        setattr(sparse, name, UnavailableSparse)
    sys.modules[sparse.__name__] = sparse
    return importlib.import_module("trellis2.models.sparse_structure_flow").SparseStructureFlowModel


def real_rope(x, phases):
    # Export phases as real [L, head_dim/2, 2] constants, not complex ONNX values.
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cosine, sine = phases[..., 0].unsqueeze(-2), phases[..., 1].unsqueeze(-2)
    even = pairs[..., 0] * cosine - pairs[..., 1] * sine
    odd = pairs[..., 0] * sine + pairs[..., 1] * cosine
    return torch.stack((even, odd), dim=-1).flatten(-2).to(x.dtype)


class TokenWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latent, timestep, cond):
        x = latent.transpose(1, 2).reshape(1, 8, 16, 16, 16)
        return self.model(x, timestep, cond).reshape(1, 8, 4096).transpose(1, 2).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint prefix, without suffix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--fixtures", type=Path, help="Captured native fixtures; otherwise deterministic synthetic tensors")
    parser.add_argument("--block-only", action="store_true", help="Export one real block as a feasibility probe")
    parser.add_argument("--rtol", type=float, default=0.02, help="Whole-network adaptation tolerance (BF16 rounding accumulates)")
    parser.add_argument("--atol", type=float, default=0.02)
    args = parser.parse_args()
    cls = load_reference(args.reference.resolve())
    config_path = Path(str(args.checkpoint) + ".json")
    weights_path = Path(str(args.checkpoint) + ".safetensors")
    config = json.loads(config_path.read_text())["args"]
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    print("Loading reference model and checkpoint", flush=True)
    model_config = dict(config)
    if args.block_only:
        model_config["num_blocks"] = 1
    model = cls(**model_config).eval()
    if args.block_only:
        with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
            weights = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()
                       if not name.startswith("blocks.") or name.startswith("blocks.0.")}
    else:
        weights = load_file(weights_path)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    del weights
    if unexpected or any(name != "rope_phases" for name in missing):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.convert_to(dtype)
    wrapper = TokenWrapper(model).eval()
    torch.manual_seed(42)
    if args.fixtures:
        sample = tuple(torch.from_numpy(np.load(args.fixtures / f"{name}.npy"))
                       for name in ["latent", "timestep", "cond"])
    else:
        sample = (torch.randn(1, 4096, 8), torch.tensor([500.0]), torch.randn(1, 1029, 1024) * 0.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixtures = args.output.parent / (args.output.stem + "_fixtures")
    fixtures.mkdir(exist_ok=True)
    # Compare the export adaptation against untouched complex-valued upstream RoPE.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper.to(device)
    gpu_sample = tuple(t.to(device) for t in sample)
    with torch.inference_mode():
        reference = wrapper(*gpu_sample).float().cpu().numpy()
    from trellis2.modules.attention.rope import RotaryPositionEmbedder
    real_phases = torch.view_as_real(model.rope_phases).clone()
    model.rope_phases = real_phases
    RotaryPositionEmbedder.apply_rotary_embedding = staticmethod(real_rope)
    with torch.inference_mode():
        adapted = wrapper(*gpu_sample).float().cpu().numpy()
    error = float(np.max(np.abs(reference - adapted)))
    if not np.allclose(reference, adapted, rtol=args.rtol, atol=args.atol):
        raise RuntimeError(f"Real-valued RoPE adaptation failed parity: max_abs={error}")
    for name, tensor in zip(["latent", "timestep", "cond"], sample):
        np.save(fixtures / f"{name}.npy", tensor.numpy())
    np.save(fixtures / "velocity.npy", reference)
    wrapper.cpu()
    del gpu_sample
    torch.cuda.empty_cache()
    # The dynamo exporter emits SDPA as standard ONNX operators; TensorRT must
    # recognize/fuse their pattern. Engine inspection is a separate acceptance gate.
    print(f"RoPE whole-network max_abs={error}; exporting {args.output}", flush=True)
    with torch.inference_mode():
        torch.onnx.export(wrapper, sample, str(args.output), dynamo=True,
                          input_names=["latent", "timestep", "cond"], output_names=["velocity"],
                          opset_version=20, external_data=True, verbose=False)
    import onnx
    onnx.checker.check_model(str(args.output))
    manifest = dict(schema_version=1, model="trellis2-ss-flow", preset=512,
                    block_only=args.block_only, precision=args.precision,
                    torch=torch.__version__, onnx=onnx.__version__,
                    reference_commit=subprocess.check_output(
                        ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip(),
                    checkpoint_sha256=sha256(weights_path), config_sha256=sha256(config_path),
                    onnx_sha256=sha256(args.output), rope_adaptation_max_abs=error,
                    adaptation_rtol=args.rtol, adaptation_atol=args.atol,
                    inputs={"latent": [1, 4096, 8], "timestep": [1], "cond": [1, 1029, 1024]},
                    outputs={"velocity": [1, 4096, 8]}, io_dtype="float32",
                    timestep_scale=1000, layout="token-major, channel contiguous",
                    fixtures=str(fixtures.resolve()),
                    external_data=[dict(file=p.name, sha256=sha256(p))
                                   for p in args.output.parent.glob(args.output.name + "*.data")])
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
