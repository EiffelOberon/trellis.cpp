"""Run the native TensorRT harness on saved references without altering them."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from validate import compare
from fetch_models import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    fixtures = sorted(path for path in args.fixtures.glob("forward_*") if path.is_dir())
    if not fixtures:
        raise RuntimeError("No reference forwards found")
    for source in fixtures:
        target = args.output / source.name
        target.mkdir(exist_ok=True)
        for name in ["latent", "timestep", "cond", "velocity"]:
            shutil.copyfile(source / f"{name}.npy", target / f"{name}.npy")
        actual = target / "trt_velocity.npy"
        actual.unlink(missing_ok=True)
        result = subprocess.run([str(args.exe.resolve()), str(args.engine.resolve()), str(target.resolve())],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (target / "native.log").write_text(result.stdout)
        if actual.exists():
            try:
                metric = compare(np.load(actual), np.load(target / "velocity.npy"), .02, .02)
                metric["passed"] = metric["passed"] and result.returncode == 0
            except RuntimeError as error:
                metric = dict(passed=False, error=str(error))
        else:
            metric = dict(passed=False, error=result.stdout)
        metric["returncode"] = result.returncode
        metrics[source.name] = metric
        print(source.name, metric, flush=True)
    report = dict(engine_sha256=sha256(args.engine), forwards=metrics,
                  passed=all(value["passed"] for value in metrics.values()))
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
