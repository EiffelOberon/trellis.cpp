"""Fail on nonfinite or out-of-tolerance ONNX predictions on captured fixtures."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(actual, reference, rtol, atol, max_rtol=0.05):
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise RuntimeError("Shape mismatch or nonfinite prediction")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    max_abs = float(np.abs(delta).max())
    relative_l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-12))
    reference_max = float(np.abs(reference).max())
    # Quantized transformer outputs cross zero. Report pointwise allclose, but
    # gate on both whole-tensor L2 and maximum error relative to signal range.
    metrics = dict(max_abs=max_abs, rmse=float(np.sqrt(np.mean(delta ** 2))),
                   relative_l2=relative_l2, relative_max=max_abs / max(reference_max, 1e-12),
                   pointwise_allclose=bool(np.allclose(actual, reference, rtol=rtol, atol=atol)),
                   rtol=rtol, atol=atol, max_rtol=max_rtol,
                   passed=bool(relative_l2 <= rtol and max_abs <= atol + max_rtol * reference_max))
    return metrics


def main():
    import onnxruntime as ort
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--max-rtol", type=float, default=0.05, help="Maximum error relative to tensor signal range; separate from L2")
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(str(args.onnx), options, providers=["CPUExecutionProvider"])
    feeds = {name: np.load(args.fixtures / f"{name}.npy") for name in ["latent", "timestep", "cond"]}
    actual = session.run(["velocity"], feeds)[0]
    metrics = compare(actual, np.load(args.fixtures / "velocity.npy"), args.rtol, args.atol, args.max_rtol)
    print(json.dumps(metrics, indent=2))
    args.fixtures.joinpath("onnx_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if not metrics["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
