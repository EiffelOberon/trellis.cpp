"""Sequential resident/streamed worker comparison, including latent/decoder parity.

Use one engine built with --weight-streaming for both runs, changing only its
runtime weight budget. A separate baseline engine can test rebuild differences.
Device-wide nvidia-smi readings and CUDA worker readings are reported separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import psutil


def run(args, name, streamed=False, profile=None):
    root = args.output / name
    root.mkdir(parents=True, exist_ok=True)
    capture = root / "capture"
    command = [str(args.exe.resolve()), str(args.image.resolve()), str((root / "asset.glb").resolve()),
               str(args.models.resolve()), "--capture-dir", str(capture.resolve())]
    command += ["--seed", str(getattr(args, 'seed', 42))]
    if profile is not None:
        if profile != "default": command += ["--profile", profile]
        if profile == "int8" and getattr(args, 'int8_engine', None):
            command += ["--ss-engine", str(args.int8_engine.resolve())]
    else:
        command += ["--profile", "native", "--ss-engine",
                    str((args.stream_engine if streamed else args.baseline_engine).resolve())]
    env = os.environ.copy()
    # Explicitly reproduce the previous decoder budgets for the control.
    for key in ("TRELLIS_BLOCK_CHUNK_MB", "TRELLIS_DECODER_LINEAR_MB", "TRELLIS_CONVNEXT_STREAM_ALL",
                "TRELLIS_CONVNEXT_STREAM_GB", "TRELLIS_DBG_ALLOC", "TRELLIS_DBG_NAN"):
        env.pop(key, None)
    if streamed and profile is None:
        command += ["--stream-flow-blocks", str(args.blocks), "--ss-weight-budget-mib", str(args.ss_budget_mib)]
    elif profile is None:
        env["TRELLIS_BLOCK_CHUNK_MB"] = "1500"
        env["TRELLIS_DECODER_LINEAR_MB"] = "1024"
    samples = []
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--loop-ms=50"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    def read():
        for line in monitor.stdout:
            try:
                samples.append(int(line.strip()) << 20)
            except ValueError:
                pass

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    time.sleep(0.3)
    idle = samples[-1] if samples else None
    peak_rss = 0
    start = time.monotonic()
    try:
        with (root / "job.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            job = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    peak_rss = max(peak_rss, job.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.02)
            code = process.wait()
    finally:
        monitor.terminate()
        monitor.wait()
        reader.join(timeout=2)
    result = dict(command=command, returncode=code, process_seconds=time.monotonic() - start,
                  peak_rss_bytes=peak_rss, device_wide_idle_bytes=idle,
                  device_wide_peak_bytes=max(samples) if samples else None,
                  worker=json.loads((root / "asset.glb.job0.json").read_text()))
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if code:
        raise RuntimeError(f"{name} failed; inspect {root / 'job.log'}")
    print(json.dumps({"variant": name, **result["worker"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--exe", type=Path, default=Path("build-baseline/trellis-job.exe"))
    parser.add_argument("--models", type=Path, default=Path("models/f16"))
    parser.add_argument("--baseline-engine", type=Path, help="Defaults to --stream-engine for a controlled same-engine comparison")
    parser.add_argument("--stream-engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--ss-budget-mib", type=int, default=1024)
    args = parser.parse_args()
    args.baseline_engine = args.baseline_engine or args.stream_engine
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(input_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  executable_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  baseline_engine_sha256=hashlib.sha256(args.baseline_engine.read_bytes()).hexdigest(),
                  stream_engine_sha256=hashlib.sha256(args.stream_engine.read_bytes()).hexdigest(),
                  gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
                  sample_ms=dict(worker_cuda=20, device_wide=50, process_rss=20),
                  note="Fresh sequential processes; seed 42, 512 preset, 12 steps, 150K faces, 1K PNG atlas. Working tree may contain uncommitted changes.")
    report["resident"] = run(args, "resident", False)
    report["streamed"] = run(args, "streamed", True)
    parity = {}
    for name in ("ss_final", "ss_logits", "shape_latent", "texture_latent", "shape_features", "texture_features"):
        a = np.load(args.output / "resident" / "capture" / f"{name}.npy")
        b = np.load(args.output / "streamed" / "capture" / f"{name}.npy")
        same_shape = a.shape == b.shape
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        d = b.astype(np.float64) - a if same_shape else None
        rel = float(np.linalg.norm(d) / max(np.linalg.norm(a.astype(np.float64)), 1e-30)) if d is not None else None
        parity[name] = dict(same_shape=same_shape, finite=finite, exact=bool(np.array_equal(a, b)),
                            relative_l2=rel, maxabs=float(np.abs(d).max()) if d is not None else None,
                            passed=bool(same_shape and finite and rel <= 1e-5))
    report["parity"] = parity
    report["parity_passed"] = all(item["passed"] for item in parity.values())
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(parity, indent=2))
    if not report["parity_passed"]:
        raise SystemExit("Output parity gate failed")


if __name__ == "__main__":
    main()
