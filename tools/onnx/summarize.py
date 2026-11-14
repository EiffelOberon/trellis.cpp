"""Collect local benchmark, parity, silhouette, and Glint evidence into one report."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root

    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8-sig"))

    baseline = read("baseline/results.json")
    tensorrt = read("tensorrt/results.json")
    fixtures = []
    for fixture in ["prop", "thin", "parts", "material"]:
        a = [x for x in baseline if x["fixture"] == fixture]
        b = [x for x in tensorrt if x["fixture"] == fixture]
        fixtures.append(dict(fixture=fixture, input_sha256=a[0]["input_sha256"],
            baseline_seconds=mean(x["seconds"] for x in a), tensorrt_seconds=mean(x["seconds"] for x in b),
            speedup=mean(x["seconds"] for x in a)/mean(x["seconds"] for x in b),
            baseline_total_peak_gib=max(x["peak_device_bytes"] for x in a)/(1<<30),
            baseline_additional_peak_gib=max(x["additional_peak_bytes"] for x in a)/(1<<30),
            tensorrt_total_peak_gib=max(x["peak_device_bytes"] for x in b)/(1<<30),
            tensorrt_additional_peak_gib=max(x["additional_peak_bytes"] for x in b)/(1<<30),
            baseline_rss_gib=max(x["peak_rss_bytes"] for x in a)/(1<<30),
            tensorrt_rss_gib=max(x["peak_rss_bytes"] for x in b)/(1<<30)))
    frame_times = {}
    for name, path in [("isolated", "glint-render/trt-lit/trellis_trt_42_frames.csv"),
                       ("generation_overlap", "glint-render/overlap/overlap_42_frames.csv")]:
        rows = list(csv.DictReader((root / path).open()))[1:] # exclude shader/first-frame setup
        frame_times[name] = {}
        for key in ["wall_frame_ms", "gpu_total_ms"]:
            values = [float(row[key]) for row in rows]
            frame_times[name][key] = dict(mean=mean(values), p95=float(np.quantile(values,.95)), maximum=max(values))
    engine = read("ss_flow.manifest.json")
    report = dict(environment=read("baseline/environment.json"), fixtures=fixtures,
        final_worker_job=read("verified.glb.job0.json"),
        engine={k:v for k,v in engine.items() if k not in ["fixtures"]},
        native_forward_vs_pytorch=read("reference/metrics.json"),
        tensorrt_forward_vs_pytorch=read("reference/trt_metrics.json"),
        silhouettes=read("silhouettes/metrics.json"),
        baseline_repeat_silhouettes=read("baseline-repeat/metrics.json"),
        glint_frame_times=frame_times,
        limits=["Device VRAM is sampled at 50 ms, not an exact allocation high-water mark.",
                "Baseline and TensorRT measurements use separate CLI processes, warm file/driver caches, and a resident Glint scene.",
                "Glint overlap measurements use a headless CUDA/OptiX renderer in another process, not the editor's UI job path.",
                "Library per-stage GPU counters use cudaMemGetInfo; WDDM values differ from device-wide nvidia-smi.",
                "One material-side silhouette is below the provisional 0.95 IoU review threshold.",
                "The original C++ sampler clamps guidance rescaling; the unmodified Microsoft sampler does not.",
                "Glint UI/MCP worker wiring and one-undo scene append remain a separate integration step."])
    output = args.output or root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(fixtures=fixtures, glint_frame_times=frame_times), indent=2))


if __name__ == "__main__":
    main()
