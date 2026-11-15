"""Strict full-trajectory geometry gate for Glint's resident CUDA profiles.

This never promotes a profile by itself: material review, SDK relocation and
integrated viewport/memory qualification are separate release requirements.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "onnx"))
from compare_silhouettes import geometry, silhouette
from validate_glb import validate


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare(reference, candidate):
    a, b = [np.load(root / "capture" / "ss_logits.npy") > 0 for root in (reference, candidate)]
    if a.shape != b.shape:
        return dict(passed=False, error="SS occupancy shapes differ")
    occupancy = float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))
    finite = all(np.isfinite(np.load(root / "capture" / (name + ".npy"))).all()
                 for root in (reference, candidate)
                 for name in ("ss_final", "ss_logits", "shape_latent", "texture_latent", "shape_features", "texture_features"))
    noise_equal = bool(np.array_equal(*(np.load(root / "capture" / "noise_0.npy") for root in (reference, candidate))))
    meshes = [geometry(root / "asset.glb") for root in (reference, candidate)]
    views = []
    for angle in (0, 90, 180, 300):
        x, y = [silhouette(*mesh, angle) for mesh in meshes]
        iou = float(np.count_nonzero(x & y) / max(1, np.count_nonzero(x | y)))
        views.append(dict(angle=angle, iou=iou, passed=iou >= .95))
    validations = [validate(root / "asset.glb") for root in (reference, candidate)]
    return dict(ss_occupancy_iou=occupancy, finite=bool(finite), identical_initial_noise=noise_equal,
                silhouettes=views, glbs=validations,
                passed=bool(occupancy >= .95 and finite and noise_equal and all(v["passed"] for v in views)
                            and all(v["valid"] for v in validations)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path("build-baseline/trellis-job.exe"))
    parser.add_argument("--models", type=Path, default=Path("models/f16"))
    parser.add_argument("--fixtures", type=Path, default=Path("docs/benchmarks/glint-qualification/fixtures.json"))
    parser.add_argument("--fixture", action="append", help="Run only named fixtures for diagnosis; not full qualification")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--int8-engine", type=Path)
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop at the first failed case while developing a profile")
    parser.add_argument("--output", type=Path, required=True, help="A new directory; existing results are never overwritten")
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    selected = [f for f in fixtures["fixtures"] if not args.fixture or f["name"] in args.fixture]
    if not selected or (args.fixture and set(args.fixture) - {f["name"] for f in selected}):
        parser.error("Unknown or empty fixture selection")
    seeds = args.seed or fixtures["seeds"]
    if any(seed < 0 or seed > 0xffffffff for seed in seeds):
        parser.error("Seeds must be unsigned 32-bit integers")
    for fixture in selected:
        if digest(Path(fixture["path"])) != fixture["sha256"]:
            raise RuntimeError("Fixture changed: " + fixture["name"])
    args.output.mkdir(parents=True, exist_ok=False)
    source_diff = subprocess.check_output(["git", "diff", "--submodule=diff", "HEAD"])
    (args.output/"source.patch").write_bytes(source_diff)
    report = dict(geometry_qualified=False, production_qualified=False,
                  full_matrix_requested=not args.fixture and set(seeds) == set(fixtures["seeds"]),
                  full_matrix_completed=False, expected_cases=len(fixtures["fixtures"])*len(fixtures["seeds"]),
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_diff_sha256=hashlib.sha256(source_diff).hexdigest(),
                  executable_sha256=digest(args.exe), fixture_manifest_sha256=digest(args.fixtures),
                  model_manifest_sha256=digest(args.models / "resident-profiles.json"),
                  gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip(),
                  cases=[])
    report["runtime_sha256"] = {name:digest(args.exe.parent/name) for name in
        ("ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll") if (args.exe.parent/name).is_file()}
    added_sources = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--",
        "cmake", "include", "src", "tools"], text=True).splitlines()
    report["untracked_source_sha256"] = {name:digest(Path(name)) for name in added_sources}
    for name in added_sources:
        destination = args.output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    if args.int8_engine:
        report["candidate_int8_engine_sha256"] = digest(args.int8_engine)
    try:
        for fixture in selected:
            for seed in seeds:
                case = dict(fixture=fixture["name"], holdout=fixture["holdout"], seed=seed, jobs={})
                report["cases"].append(case)
                roots = {}
                for profile in ("native", "bf16", "int8"):
                    if shutil.disk_usage(args.output).free < 2 << 30:
                        raise RuntimeError('Qualification stopped with less than 2 GiB free disk space')
                    root = args.output / f"{fixture['name']}-{seed}" / profile
                    root.mkdir(parents=True)
                    roots[profile] = root
                    command = [str(args.exe.resolve()), str(Path(fixture["path"]).resolve()),
                               str((root / "asset.glb").resolve()), str(args.models.resolve()),
                               "--profile", profile, "--seed", str(seed), "--capture-dir", str((root / "capture").resolve())]
                    if profile == "int8" and args.int8_engine:
                        command += ["--ss-engine", str(args.int8_engine.resolve())]
                    with (root / "job.log").open("w", encoding="utf-8") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    metrics = json.loads((root / "asset.glb.job0.json").read_text())
                    if metrics["profile"] != profile or metrics["seed"] != seed or metrics["repaired_nonfinite"] != 0:
                        raise RuntimeError("Unexpected profile, seed or nonfinite repair")
                    case["jobs"][profile] = metrics
                case["bf16_vs_native"] = compare(roots["native"], roots["bf16"])
                case["int8_vs_bf16"] = compare(roots["bf16"], roots["int8"])
                case["passed"] = case["bf16_vs_native"]["passed"] and case["int8_vs_bf16"]["passed"]
                print(json.dumps({k:v for k,v in case.items() if k != "jobs"}), flush=True)
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                if args.stop_on_failure and not case["passed"]:
                    raise SystemExit('Stopped at the first failed geometry case')
        report["full_matrix_completed"] = report["full_matrix_requested"] and len(report["cases"]) == report["expected_cases"]
        report["geometry_qualified"] = report["full_matrix_completed"] and all(c["passed"] for c in report["cases"])
    finally:
        report["completed_cases"] = sum("passed" in case for case in report["cases"])
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["geometry_qualified"]:
        raise SystemExit("Full production geometry qualification has not passed")


if __name__ == "__main__":
    main()
