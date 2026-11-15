"""Run reproducible CLI fixtures with sampled device-wide VRAM and process RSS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import psutil

FIXTURES = {
    "prop": "assets/showcase/chest/chest.png",
    "thin": "assets/showcase/axe/axe.png",
    "parts": "assets/showcase/drone/drone.png",
    "material": "assets/showcase/knight/knight.png",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("models/f16"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", choices=list(FIXTURES), action="append")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--capture-root", type=Path, help="Optional per-run parity captures (included in timing)")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                   "--format=csv,noheader"], text=True).strip(),
                      sample_interval_ms=50, seed=42, resolution=512, atlas=1024, faces=150000,
                      note="Repeated process launches: OS/file cache warm, models reloaded each run.")
    (args.output / "environment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    results = []
    for fixture in args.fixture or list(FIXTURES):
        image = Path(FIXTURES[fixture])
        for run in range(args.repeats):
            stem = f"{fixture}_{run}"
            command = [str(args.exe.resolve()), str(image.resolve()), str((args.output / f"{stem}.glb").resolve()),
                       "--models", str(args.models.resolve()), "--seed", "42", "--res", "512",
                       "--atlas", "1024", "--faces", "150000", "--webp", "off", "--require-gpu"]
            if args.engine:
                command += ["--ss-engine", str(args.engine.resolve())]
            if args.capture_root:
                command += ["--capture-dir", str((args.capture_root / stem).resolve())]
            samples = []
            monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                        "--loop-ms=50"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

            def read_gpu():
                for line in monitor.stdout:
                    try:
                        samples.append(int(line.strip()) * (1 << 20))
                    except ValueError:
                        pass

            reader = threading.Thread(target=read_gpu, daemon=True)
            reader.start()
            time.sleep(0.3)
            idle = samples[-1] if samples else None
            start = time.monotonic()
            peak_rss = 0
            with (args.output / f"{stem}.log").open("w") as log:
                job = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                process = psutil.Process(job.pid)
                while job.poll() is None:
                    try:
                        peak_rss = max(peak_rss, process.memory_info().rss)
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(0.05)
                code = job.wait()
            monitor.terminate()
            monitor.wait()
            reader.join(timeout=2)
            record = dict(fixture=fixture, run=run, command=command, returncode=code,
                          input_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                          seconds=time.monotonic() - start, peak_rss_bytes=peak_rss,
                          idle_device_bytes=idle, peak_device_bytes=max(samples) if samples else None,
                          additional_peak_bytes=max(samples) - idle if samples and idle is not None else None)
            results.append(record)
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            if code:
                raise SystemExit(code)


if __name__ == "__main__":
    main()
