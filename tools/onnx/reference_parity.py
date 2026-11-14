"""Replay captured native SS inputs through Microsoft's unmodified dense model/sampler."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from safetensors.torch import load_file
from export_ss_flow import load_reference, TokenWrapper
from validate import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()
    cls = load_reference(args.reference.resolve())
    config = json.loads(Path(str(args.checkpoint) + ".json").read_text())["args"]
    model = cls(**config).eval()
    missing, unexpected = model.load_state_dict(load_file(str(args.checkpoint) + ".safetensors"), strict=False)
    if unexpected or any(name != "rope_phases" for name in missing):
        raise RuntimeError(f"Checkpoint mismatch: {missing}, {unexpected}")
    model.convert_to({"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision])
    model.cuda()
    wrapper = TokenWrapper(model).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    directories = sorted(args.capture.glob("forward_*"))
    if not directories:
        raise RuntimeError("No captured forwards")
    metrics = {}
    with torch.inference_mode():
        for index in sorted({0, 1, len(directories)//2, len(directories)-1}):
            directory = directories[index]
            inputs = [torch.from_numpy(np.load(directory / f"{name}.npy")).cuda()
                      for name in ["latent", "timestep", "cond"]]
            actual = wrapper(*inputs).float().cpu().numpy()
            np.save(args.output / f"{directory.name}.npy", actual)
            fixture = args.output / directory.name
            fixture.mkdir(exist_ok=True)
            for name, tensor in zip(["latent", "timestep", "cond"], inputs):
                np.save(fixture / f"{name}.npy", tensor.cpu().numpy())
            np.save(fixture / "velocity.npy", actual)
            metrics[directory.name] = compare(np.load(directory / "velocity.npy"), actual, .02, .02)
            print(directory.name, metrics[directory.name], flush=True)
        # Bootstrap only the sampler package; root pipelines imports sparse decoders.
        pipelines = types.ModuleType("trellis2.pipelines")
        pipelines.__path__ = [str(args.reference.resolve() / "trellis2/pipelines")]
        sys.modules[pipelines.__name__] = pipelines
        sampler_cls = importlib.import_module("trellis2.pipelines.samplers").FlowEulerGuidanceIntervalSampler
        noise = np.load(args.capture / "noise_0.npy").reshape(1,4096,8).transpose(0,2,1).reshape(1,8,16,16,16).copy()
        cond = torch.from_numpy(np.load(directories[0] / "cond.npy")).cuda()
        neg = torch.zeros_like(cond)
        sampled = sampler_cls(sigma_min=1e-5).sample(
            model, torch.from_numpy(noise).cuda(), cond=cond, neg_cond=neg, steps=12,
            rescale_t=5., guidance_strength=7.5, guidance_rescale=.7, guidance_interval=(.6,1.), verbose=False)
        latent = sampled.samples.reshape(1,8,4096).transpose(1,2).contiguous().float().cpu().numpy()
        np.save(args.output / "ss_final.npy", latent)
        metrics["trajectory"] = compare(np.load(args.capture / "ss_final.npy"), latent, .02, .02)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
