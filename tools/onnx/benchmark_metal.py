"""Measure the fixed Metal worker workload and compare occupancy and GLB silhouettes."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from compare_silhouettes import geometry, silhouette
from fetch_models import sha256
from validate_glb import validate

FIXTURES = ("axe", "chest", "drone", "knight")


def system_memory():
    commands = {"swap": ["sysctl", "-n", "vm.swapusage"], "vm_stat": ["vm_stat"],
                "pressure": ["memory_pressure", "-Q"]}
    return {key: subprocess.check_output(command, text=True).strip() for key, command in commands.items()}


def measure(args, fixture, backend, weights):
    image = Path(f"assets/showcase/{fixture}/{fixture}.png").resolve()
    directory = (args.output / fixture / backend).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    runs = []
    for label, count in [("cold", 1), ("warm", args.repeats + 1)]:
        stem = directory / label
        command = [str(args.exe.resolve()), str(image), f"{stem}.glb", str(args.models.resolve()),
                   "--device", "metal", "--ss-backend", backend, "--repeat", str(count)]
        if weights:
            command.extend(["--ss-weights", str(weights.resolve())])
        if label == "cold":
            command.extend(["--capture-dir", str(directory / "capture")])
        before = system_memory()
        start = time.monotonic()
        with (directory / f"{label}.log").open("w") as log:
            job = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        launch_seconds = time.monotonic() - start
        after = system_memory()
        evidence = dict(command=command, returncode=job.returncode, launch_seconds=launch_seconds,
                        memory_before=before, memory_after=after)
        (directory / f"{label}-system.json").write_text(json.dumps(evidence, indent=2) + "\n")
        job.check_returncode()
        for i in range(count):
            asset = Path(f"{stem}.glb" if i == 0 else f"{stem}.glb.job{i}.glb")
            metrics = json.loads(Path(f"{stem}.glb.job{i}.json").read_text())
            if metrics["status"] != "success":
                raise RuntimeError(metrics["error"])
            glb = validate(asset)
            if any(tuple(size) != (1024, 1024) for size in glb["textures"]):
                raise RuntimeError(f"Expected 1024 PBR atlas: {asset}")
            additional = metrics["sampled_peak_footprint_bytes"] - metrics["idle_footprint_bytes"]
            runs.append(dict(mode=label, iteration=i, additional_footprint_bytes=additional,
                             metrics=metrics, glb=glb))
    timed = [run["metrics"]["seconds"] for run in runs if run["mode"] == "warm" and run["iteration"] > 0]
    spread = dict(median=statistics.median(timed), min=min(timed), max=max(timed))
    return dict(image_sha256=sha256(image), jobs=runs, warm_seconds=spread)


def iou(a, b):
    union = np.count_nonzero(a | b)
    if not union:
        raise RuntimeError("Empty occupancy or silhouettes")
    return float(np.count_nonzero(a & b) / union)


def quality(ggml, mlx):
    ggml_a = Path(ggml["jobs"][0]["glb"]["path"])
    ggml_b = Path(ggml["jobs"][1]["glb"]["path"])
    mlx_a = Path(mlx["jobs"][0]["glb"]["path"])
    geometries = [geometry(asset) for asset in [ggml_a, ggml_b, mlx_a]]
    silhouettes = []
    for angle in [0, 90, 180, 300]:
        a, b, c = [silhouette(*mesh, angle) for mesh in geometries]
        silhouettes.append(dict(angle=angle, repeat_iou=iou(a, b), mlx_iou=iou(a, c)))
    a, b = [np.load(asset.parent / "capture" / "ss_final.npy").astype(np.float64) for asset in [ggml_a, mlx_a]]
    relative_l2 = float(np.linalg.norm(a-b) / np.linalg.norm(a))
    a, b = [np.load(asset.parent / "capture" / "ss_logits.npy") > 0 for asset in [ggml_a, mlx_a]]
    occupancy = iou(a, b)
    return dict(final_latent_relative_l2=relative_l2, occupancy_iou=occupancy, silhouettes=silhouettes,
                passed=occupancy >= .95 and all(view["mlx_iou"] >= .95 for view in silhouettes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True, help="trellis-job executable")
    parser.add_argument("--models", type=Path, default=Path("models/f16"))
    parser.add_argument("--weights", type=Path, help="Include MLX and compare against GGML")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", choices=FIXTURES, action="append")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("At least one warm repetition is required")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"fixtures": {}}
    for fixture in args.fixture or FIXTURES:
        results = {"ggml": measure(args, fixture, "ggml", None)}
        if args.weights:
            results["mlx"] = measure(args, fixture, "mlx", args.weights)
            results["quality"] = quality(results["ggml"], results["mlx"])
        summary["fixtures"][fixture] = results
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(result.get("quality", {}).get("passed", True) for result in summary["fixtures"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
