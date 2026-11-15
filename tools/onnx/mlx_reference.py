"""Replay real SS captures against the pinned, unmodified Microsoft model."""
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
from safetensors import safe_open

from export_ss_flow import load_reference, TokenWrapper
from fetch_models import sha256
from validate import compare

REFERENCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
CHECKPOINT_HASH = "ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6"
CONFIG_HASH = "6cdfb636854f60ffcadd8c49244169cd449f34c5ccdb55b5ff7c04678f9c399e"
INPUT_NAMES = ("latent", "timestep", "cond")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Original safetensors file")
    parser.add_argument("--capture", type=Path, required=True, help="Native --capture-dir")
    parser.add_argument("--probe", type=Path, required=True, help="trellis-probe-mlx executable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--reference-device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--block", action="store_true")
    parser.add_argument("--native-latent", action="store_true", help="Also run the original sampler policy")
    args = parser.parse_args()
    if args.block and args.native_latent:
        parser.error("A single block cannot produce a reference trajectory")

    revision = subprocess.check_output(["git", "-C", args.reference, "rev-parse", "HEAD"], text=True).strip()
    if revision != REFERENCE_REVISION:
        raise RuntimeError(f"Unexpected upstream source revision {revision}")
    config_path = args.checkpoint.with_suffix(".json")
    if sha256(args.checkpoint) != CHECKPOINT_HASH or sha256(config_path) != CONFIG_HASH:
        raise RuntimeError("Pinned SS checkpoint or config hash mismatch")
    inputs = sorted(args.capture.glob("forward_*"))
    if len(inputs) < 4:
        raise RuntimeError("Capture must include early, negative, middle and late forwards")

    config = json.loads(config_path.read_text())["args"]
    if args.block:
        config["num_blocks"] = 1
    cls = load_reference(args.reference.resolve())
    model = cls(**config).eval()
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        weights = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()
                   if not args.block or not key.startswith("blocks.") or key.startswith("blocks.0.")}
        missing, unexpected = model.load_state_dict(weights, strict=False)
    del weights
    if unexpected or missing != ["rope_phases"]:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    model.convert_to(dtype)
    wrapper = TokenWrapper(model).to(args.reference_device)
    outputs = {}
    if args.block:
        def hook(name):
            def save(_module, _inputs, output):
                outputs[name] = output.detach().float().cpu().numpy()
            return save
        block = model.blocks[0]
        for name, module in [("self", block.self_attn), ("cross", block.cross_attn),
                             ("ff", block.mlp), ("block", block)]:
            module.register_forward_hook(hook(name))

    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = [inputs[0]] if args.block else [inputs[0], inputs[1], inputs[len(inputs)//2], inputs[-1]]
    metrics = {}
    for fixture in fixtures:
        output = args.output / fixture.name
        output.mkdir(exist_ok=True)
        with (output / "mlx.log").open("w") as log:
            subprocess.run([args.probe.resolve(), args.checkpoint.resolve(), fixture.resolve(), output.resolve(),
                            args.precision, "block" if args.block else "full"], check=True, stdout=log,
                           stderr=subprocess.STDOUT)
        feeds = [torch.from_numpy(np.load(fixture / f"{name}.npy")).to(args.reference_device)
                 for name in INPUT_NAMES]
        with torch.inference_mode():
            velocity = wrapper(*feeds).float().cpu().numpy()
        if not args.block:
            outputs = {"velocity": velocity}
        for name, reference in outputs.items():
            np.save(output / f"reference_{name}.npy", reference)
            actual = np.load(output / f"{name}.npy")
            stats = compare(actual, reference, .02, .02)
            stats["pointwise_outliers"] = int(np.count_nonzero(~np.isclose(actual, reference, rtol=.02, atol=.02)))
            metrics[f"{fixture.name}/{name}"] = stats
            print(fixture.name, name, stats, flush=True)
        if not args.block:
            metrics[f"{fixture.name}/ggml"] = compare(np.load(fixture / "velocity.npy"), velocity, .02, .02)

    if args.native_latent:
        pipelines = types.ModuleType("trellis2.pipelines")
        pipelines.__path__ = [str(args.reference.resolve() / "trellis2/pipelines")]
        sys.modules[pipelines.__name__] = pipelines
        sampler = importlib.import_module("trellis2.pipelines.samplers").FlowEulerGuidanceIntervalSampler
        noise = np.load(args.capture / "noise_0.npy").reshape(1,4096,8).transpose(0,2,1).reshape(1,8,16,16,16).copy()
        cond = torch.from_numpy(np.load(inputs[0] / "cond.npy")).to(args.reference_device)
        with torch.inference_mode():
            latent = sampler(sigma_min=1e-5).sample(
                model, torch.from_numpy(noise).to(args.reference_device), cond=cond, neg_cond=torch.zeros_like(cond),
                steps=12, rescale_t=5., guidance_strength=7.5, guidance_rescale=.7,
                guidance_interval=(.6,1.), verbose=False).samples
        latent = latent.reshape(1,8,4096).transpose(1,2).contiguous().float().cpu().numpy()
        np.save(args.output / "reference_ss_final.npy", latent)
        metrics["trajectory_original_sampler"] = compare(np.load(args.capture / "ss_final.npy"), latent, .02, .02)

    manifest = dict(reference_commit=revision, source_weights_sha256=CHECKPOINT_HASH,
                    source_config_sha256=CONFIG_HASH, torch=torch.__version__, numpy=np.__version__,
                    precision=args.precision, reference_device=args.reference_device,
                    mlx_enable_tf32=os.environ.get("MLX_ENABLE_TF32"),
                    capture={str(p.relative_to(args.capture)): sha256(p)
                             for p in sorted(args.capture.rglob("*.npy"))},
                    results={str(p.relative_to(args.output)): sha256(p)
                             for p in sorted(args.output.rglob("*.npy"))},
                    metrics=metrics)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if all(m["passed"] for k, m in metrics.items()
                    if k.startswith("forward_") and not k.endswith("/ggml")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
