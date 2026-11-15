"""Use BF16 fake quantization to screen INT8 policies before native engine builds.

Screening is approximate: accepted candidates still require actual TensorRT
execution because integer accumulation and fusion differ from BF16 matmuls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from export_ss_flow import load_reference, TokenWrapper, real_rope
from validate import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", action="append", help="Run only the named policy; repeatable")
    parser.add_argument("--late-only", action="store_true", help="Screen only the final forward")
    args = parser.parse_args()
    cls = load_reference(args.reference.resolve())
    config = json.loads(Path(str(args.checkpoint) + ".json").read_text())["args"]
    model = cls(**config).eval()
    missing, unexpected = model.load_state_dict(load_file(str(args.checkpoint) + ".safetensors"), strict=False)
    if unexpected or any(name != "rope_phases" for name in missing):
        raise RuntimeError("Checkpoint mismatch")
    model.convert_to(torch.bfloat16)
    from trellis2.modules.attention.rope import RotaryPositionEmbedder
    model.rope_phases = torch.view_as_real(model.rope_phases).clone()
    RotaryPositionEmbedder.apply_rotary_embedding = staticmethod(real_rope)
    wrapper = TokenWrapper(model).eval().cuda()
    maxima = json.loads(args.scales.read_text())["channels"]
    modules = {name: module for name, module in wrapper.named_modules() if name in maxima}
    originals = {name: module.weight.detach().cpu().clone() for name, module in modules.items()}
    fixtures = [path for path in sorted(args.fixtures.glob("forward_*")) if path.is_dir()]
    if args.late_only:
        fixtures = fixtures[-1:]
    policies = [
        ("all-a035", .35, [], []), ("all-a065", .65, [], []),
        ("no-fc2", .5, [], ["mlp.mlp.2"]),
        ("attention-only", .5, ["attn."], []),
        ("input-projections", .5, [], ["mlp.mlp.2", "to_out"]),
        ("self-attn-only", .5, ["self_attn."], []),
        ("cross-kv-only", .5, ["cross_attn.to_kv"], []),
        ("all-a08", .8, [], []),
        ("no-fc2-a065", .65, [], ["mlp.mlp.2"]),
        ("attention-a065", .65, ["attn."], []),
        ("self-plus-kv", .5, ["self_attn.", "cross_attn.to_kv"], []),
        ("no-last5", .65, [], [f"blocks.{i}." for i in range(25,30)]),
        ("no-last10", .65, [], [f"blocks.{i}." for i in range(20,30)]),
    ]
    if args.policy:
        unknown = set(args.policy) - {policy[0] for policy in policies}
        if unknown:
            parser.error(f"Unknown policy: {unknown}")
        policies = [policy for policy in policies if policy[0] in args.policy]
    print("Example modules:", list(modules)[:10], flush=True)
    report = {}
    with torch.inference_mode():
        for label, alpha, include, exclude in policies:
            handles, weights, count = [], 0, 0
            for name, module in modules.items():
                module.weight.copy_(originals[name])
                if (include and not any(p in name for p in include)) or any(p in name for p in exclude):
                    continue
                w = module.weight.float()
                activation = torch.tensor(maxima[name], device="cuda")
                smooth = (activation.clamp_min(1e-5).pow(alpha) /
                          w.abs().amax(dim=0).clamp_min(1e-5).pow(1-alpha)).clamp(1e-3, 1e3).bfloat16().float()
                w *= smooth.unsqueeze(0)
                ws = (w.abs().amax(dim=1) / 127).clamp_min(1e-10).bfloat16().float()
                xscale = ((activation / smooth).amax() / 127).clamp_min(1e-10).bfloat16().float()
                module.weight.copy_((w / ws[:, None]).round().clamp(-127,127) * ws[:, None])

                def hook(layer, inputs, s=smooth, scale=xscale):
                    x = inputs[0].float() / s
                    x = (x / scale).round().clamp(-127,127) * scale
                    return (x.bfloat16(), *inputs[1:])

                handles.append(module.register_forward_pre_hook(hook))
                count += 1
                weights += module.weight.numel()
            results = {}
            for fixture in fixtures:
                inputs = [torch.from_numpy(np.load(fixture / f"{name}.npy")).cuda()
                          for name in ["latent", "timestep", "cond"]]
                actual = wrapper(*inputs).float().cpu().numpy()
                results[fixture.name] = compare(actual, np.load(fixture / "velocity.npy"), .02, .02)
            for handle in handles:
                handle.remove()
            report[label] = dict(alpha=alpha, include=include, exclude=exclude, layers=count,
                                 weights=weights, forwards=results,
                                 passed=all(x["passed"] for x in results.values()))
            print(label, "layers", count, "L2", [round(x["relative_l2"],5) for x in results.values()],
                  "passed", report[label]["passed"], flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
