"""Check engine inspection evidence for fused self/cross attention and bounded scratch.

This verifies the fixed 4096-token export only; it is not a generic proof for
other graphs, dynamic profiles, plugins, or GPUs.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    args = parser.parse_args()
    manifest_path = args.engine.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    layers = json.loads(args.engine.with_suffix(".layers.json").read_text())["Layers"]
    attention = [layer for layer in layers if "mha" in layer.get("Name", "").lower()]
    expected = 2 if manifest["block_only"] else 60
    # The smallest full score tensor for this model is [1,12,4096,4096]
    # in BF16/F16 (384 MiB). FP32 is 768 MiB. Entire context scratch is below it.
    score_bytes = 12 * 4096 * 4096 * (4 if manifest["precision"] == "fp32" else 2)
    passed = len(attention) == expected and manifest["context_device_memory_bytes"] < score_bytes
    evidence = dict(fused_attention_layers=len(attention), expected_attention_layers=expected,
                    full_self_attention_score_bytes=score_bytes,
                    context_scratch_bytes=manifest["context_device_memory_bytes"], passed=passed,
                    method="TensorRT engine layer inspection plus total context-scratch bound",
                    scope="fixed B=1, 4096-token dense SS model; standard TensorRT operators")
    manifest["attention_memory_verified"] = passed
    manifest["attention_evidence"] = evidence
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
