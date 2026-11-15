"""Collect the INT8 experiments, paired job measurements, and quality checks."""
import argparse
import json
from pathlib import Path
import re
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--bf16-engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))

    engines = {}
    for label, stem in [("minmax", "ss_flow_int8"), ("smooth_all", "ss_flow_int8_smooth"),
                        ("mixed", "ss_flow_int8_mixed")]:
        path = args.root / stem
        manifest = read(path.with_suffix(".manifest.json"))
        layers = read(path.with_suffix(".layers.json"))["Layers"]
        quantization = manifest["quantization"]
        evaluation = {"minmax": "evaluation", "smooth_all": "evaluation-smooth", "mixed": "evaluation-mixed"}[label]
        engines[label] = dict(engine_bytes=path.with_suffix(".plan").stat().st_size,
            engine_sha256=manifest["engine_sha256"], context_scratch_bytes=manifest["context_device_memory_bytes"],
            onnx_sha256=manifest["onnx_sha256"], external_data=manifest["external_data"],
            source_checkpoint_sha256=manifest["checkpoint_sha256"], tensorrt=manifest["tensorrt"], gpu=manifest["gpu"],
            calibration_sha256=manifest["calibration_sha256"], calibration_forwards=len(quantization["fixtures"]),
            quantized_linears=len(quantization["layers"]), quantized_weights=sum(x["weights"] for x in quantization["layers"]),
            smooth_alpha=quantization["smooth_alpha"],
            int8_input_layers=sum(any(i.get("Datatype") == "Int8" for i in x.get("Inputs", [])) for x in layers),
            attention=manifest["attention_evidence"], forwards=read(args.root / evaluation / "metrics.json"))
    bf16 = read(args.bf16_engine.with_suffix(".manifest.json"))
    paired = []
    baseline = read(args.root / "bf16/results.json")
    mixed = read(args.root / "mixed/results.json")
    for a, b in zip(baseline, mixed, strict=True):
        if a["fixture"] != b["fixture"] or a["input_sha256"] != b["input_sha256"]:
            raise RuntimeError("Benchmark fixture mismatch")
        flow_times = {}
        for label in ["bf16", "mixed"]:
            log = (args.root / label / f"{a['fixture']}_0.log").read_text()
            flow_times[label] = [float(x) for x in re.findall(r"\[flow\] \d+ steps, \d+ forwards, ([0-9.]+)s", log)]
        paired.append(dict(fixture=a["fixture"], input_sha256=a["input_sha256"],
            bf16_seconds=a["seconds"], mixed_seconds=b["seconds"], speedup=a["seconds"]/b["seconds"],
            bf16_additional_peak_gib=a["additional_peak_bytes"]/(1<<30),
            mixed_additional_peak_gib=b["additional_peak_bytes"]/(1<<30),
            bf16_total_peak_gib=a["peak_device_bytes"]/(1<<30), mixed_total_peak_gib=b["peak_device_bytes"]/(1<<30),
            bf16_rss_gib=a["peak_rss_bytes"]/(1<<30), mixed_rss_gib=b["peak_rss_bytes"]/(1<<30),
            flow_seconds=flow_times))
        capture_a = args.root / "bf16-captures" / f"{a['fixture']}_0"
        capture_b = args.root / "mixed-captures" / f"{a['fixture']}_0"
        latent_a, latent_b = [np.load(path / "ss_final.npy").astype(np.float64) for path in [capture_a, capture_b]]
        paired[-1]["ss_final_relative_l2"] = float(np.linalg.norm(latent_a-latent_b) / np.linalg.norm(latent_a))
        paired[-1]["identical_initial_noise"] = bool(np.array_equal(np.load(capture_a / "noise_0.npy"), np.load(capture_b / "noise_0.npy")))
    occupancy = {}
    for fixture in ["prop", "thin", "parts", "material"]:
        log = (args.root / f"occupancy-{fixture}.log").read_text()
        occupancy[fixture] = json.loads(next(line for line in log.splitlines() if line.startswith('{"occupancy_iou"')))
    silhouettes = read(args.root / "silhouettes/metrics.json")
    quality_passed = all(value["occupancy_iou"] >= .95 for value in occupancy.values()) and all(x["passed_iou_95"] for x in silhouettes)
    report = dict(status="experimental", quality_passed=quality_passed,
        bf16_engine_sha256=bf16["engine_sha256"],
        bf16_engine_bytes=args.bf16_engine.stat().st_size, engines=engines, paired_jobs=paired,
        environment=read(args.root / "bf16/environment.json"),
        silhouettes=silhouettes, occupancy=occupancy,
        worker_jobs=[read(path) for path in sorted(args.root.glob("mixed-worker.glb.job*.json"))],
        glb_validation=read(args.root / "glb-validation.json"),
        limits=["Experimental mixed INT8/BF16 SS only; remaining pipeline stages retain their original precision.",
                "Calibration uses 66 forwards from chest/drone/knight at seed 42; this is not broad production calibration.",
                "Numerical policy screening used the saved reference forwards; these are development checks, not an untouched final test set.",
                "Paired timings are one process launch per fixture/backend, with parity-capture overhead included in both.",
                "Device-wide memory is sampled at 50 ms; observed peaks are lower bounds.",
                "No in-process Glint editor generation/upload/acceleration-build acceptance was performed."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(paired_jobs=paired, occupancy=occupancy), indent=2))


if __name__ == "__main__":
    main()
