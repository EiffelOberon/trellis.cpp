# INT8 TensorRT experiment on RTX 5080

This page records the original SS-only experiment. The follow-up
[resident precision profiles](resident-profiles.md) keep BF16 as the default
and additionally quantize native shape/texture flow weights in the INT8 option.
Both profiles keep weights on the GPU and disable weight streaming.

Status: **runs end to end, but fails the current quality acceptance gates**.
Keep BF16 as the reference TensorRT configuration. INT8 remains an explicitly
selected experiment; no default backend or model precision has changed.

This experiment starts from `45edf32c2c5031932b735a9db38accba8e03d704` on
`codex/int8-ss-flow`. The CUDA/TensorRT/checkpoint environment is the same as the
[BF16 validation report](onnx-tensorrt.md). Results and checks are collected in
the [machine-readable summary](benchmarks/int8-tensorrt-5080/summary.json).
Local graphs, engines, calibration captures, GLBs, and raw logs are in `out/int8`.

## What was quantized

The first attempt quantized all 210 SS transformer linear layers with symmetric
INT8 activations and per-output-channel INT8 weights. Input activation scales
were calibrated by replaying 66 captured forwards: chest, drone, and knight,
12 sampler steps per image, including positive and negative conditioning.
The actual checkpoint and calibration inputs are hashed in the manifests.

Attention, normalization, time/input/output projections, and public FP32 I/O
retain their original precision. Calibration and conversion run offline;
the native runtime still requires no Python. This is explicit ONNX Q/DQ
quantization rather than an implicit TensorRT builder precision flag.
[TensorRT quantization documentation](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/inference-library/work-with-quantized-types.html)

Plain min/max calibration failed the forward-error gate. SmoothQuant-style
input/weight balancing improved it, but full transformer quantization still
failed the late-step check. Approximate BF16 fake-quantization screening then
identified a mixed policy for actual engine testing:

- Quantize the first 20 blocks: **140 linear layers, 849,346,560 weights**.
- Keep the final 10 blocks at their original precision.
- Use balancing exponent 0.65, rounded BF16 scales, symmetric INT8 quantization,
  and per-tensor activation / per-output-channel weight scaling.

The final engine is **1,795,295,660 bytes**, versus BF16's 2,632,776,252 bytes:
about **32% smaller**. Its SHA-256 is
`8a67f953f452bb2f919e94468579da5af8fde55a7e353e740fbda9d1f21c0bcd`.
Inspection finds 140 layers consuming INT8 inputs, all 60 fused attention
operations, and 183,361,536 bytes of context scratch. It satisfies the existing
fixed-graph attention-memory check. The layer records include INT8 weights;
this is actual low-precision execution rather than file compression alone.

## Forward checks and complete-job quality

The same early, negative-conditioning, middle, and late inputs were compared
with the saved original Microsoft PyTorch predictions:

| Policy | Early L2 | Negative L2 | Middle L2 | Late L2 | Forward gate |
| --- | ---: | ---: | ---: | ---: | --- |
| All linears, min/max | 5.47% | 4.80% | 6.27% | 11.69% | Fail |
| All linears, balancing 0.5 | 1.84% | 1.79% | 1.93% | 4.89% | Fail |
| First 20 blocks, balancing 0.65 | 0.89% | 0.76% | 0.79% | 1.99% | Pass |

The gate is unchanged: relative L2 <= 2% and max absolute error <=
`0.02 + 0.05 * max(abs(reference))`. The accepted mixed engine does not pass
pointwise allclose for every element. Axe was excluded from scale calibration,
but its saved reference forwards were used during policy selection; these are
development checks, not a final untouched test set.

Passing isolated forwards did not guarantee a matching complete trajectory.
Fresh BF16 and mixed jobs used identical actual SS initial noise for each image.
Their native SS decodes and four-view silhouettes compare as follows:

| Fixture | SS occupancy IoU | Final SS latent relative L2 | Minimum silhouette IoU |
| --- | ---: | ---: | ---: |
| Chest | **91.15%** | 42.00% | 97.97% |
| Axe | 98.80% | 12.56% | 96.85% |
| Drone | 96.82% | 15.10% | 96.79% |
| Knight | 95.42% | 15.73% | **94.97%** |

Chest occupancy falls below 95%, and one knight side silhouette falls below 95%.
These failures remain failures; thresholds were not relaxed. All eight BF16/mixed
GLBs passed structural/PBR checks, and all four mixed assets imported through
the existing Glint IO library with both embedded textures and PBR bindings.
That establishes valid assets, not final material/visual approval.

## Full-pipeline memory and time

Fresh paired runs used seed 42, resolution 512, 12 steps, 150,000 target faces,
and a 1024 PNG PBR atlas. There was one launch per fixture/backend, with capture
overhead included equally. These timings indicate a possible benefit; they are
not a multi-run statistical performance claim. Changed generated geometry also
changes downstream decode/mesh/bake work.

| Fixture | BF16 seconds | Mixed seconds | BF16 extra peak GiB | Mixed extra peak GiB |
| --- | ---: | ---: | ---: | ---: |
| Chest | 65.11 | 62.19 | 3.546 | 3.573 |
| Axe | 21.53 | 20.08 | 3.214 | 2.897 |
| Drone | 42.52 | 41.42 | 3.284 | 3.286 |
| Knight | 25.56 | 24.06 | 3.214 | 3.001 |

The SS sampling loop fell from about 3.3 seconds to about 2.3 seconds; see the
summary for each recorded loop time. Whole-job time fell by roughly 3–7% in
these single runs. Only axe and knight showed meaningful full-job VRAM savings,
about **0.32 and 0.21 GiB** respectively. Chest and drone were slightly higher.

Device-wide peaks were sampled with `nvidia-smi` every 50 ms and are lower
bounds. Idle device usage in this session was about 2.63 GiB; this differs from
the earlier BF16 session's desktop/Glint baseline. Use the paired additional
peaks, not an absolute-total comparison between sessions.

The native mixed axe worker reports SS-stage usage around 3.33 GiB through
`cudaMemGetInfo`, while shape and texture flow reach about 3.94 GiB. Later model
stages now set the peak for that asset. These counters have different accounting
from the device-wide samples and must not be combined arithmetically.
Two consecutive worker jobs succeeded with no nonfinite repairs and returned
to exactly 1,596,391,424 bytes idle usage. CUDA and CPU contract tests passed.

## Reproduce

First build the optional TensorRT runtime and obtain the pinned BF16 graph and
checkpoint using [the existing instructions](onnx-tensorrt.md). Runtime DLLs
must be on `PATH`. Capture each calibration asset with `trellis-job` and the
BF16 engine, placing captures under `out/int8/calibration/ASSET`.

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/onnx/quantize_int8.py `
  --onnx out/onnx-tensorrt/ss_flow.onnx --reference out/TRELLIS.2 `
  --checkpoint models/reference/ckpts/ss_flow_img_dit_1_3B_64_bf16 `
  --calibration out/int8/calibration --scales out/int8/calibration-scales.json `
  --smooth-alpha 0.65 `
  --exclude blocks.20. --exclude blocks.21. --exclude blocks.22. `
  --exclude blocks.23. --exclude blocks.24. --exclude blocks.25. `
  --exclude blocks.26. --exclude blocks.27. --exclude blocks.28. `
  --exclude blocks.29. --output out/int8/ss_flow_int8_mixed.onnx
.\.venv\Scripts\python.exe -X utf8 tools/onnx/build_engine.py `
  out/int8/ss_flow_int8_mixed.onnx out/int8/ss_flow_int8_mixed.plan
.\.venv\Scripts\python.exe -X utf8 tools/onnx/inspect_attention.py `
  out/int8/ss_flow_int8_mixed.plan
.\.venv\Scripts\python.exe -X utf8 tools/onnx/evaluate_engine.py `
  --exe build-baseline/trellis-test-trt.exe `
  --engine out/int8/ss_flow_int8_mixed.plan `
  --fixtures out/onnx-tensorrt/reference --output out/int8/evaluation-mixed
```

Choose a fresh output filename to rerun conversion; it refuses to overwrite an
existing graph/weight file. Omit exclusions for the all-linear experiment;
omit `--smooth-alpha` for plain min/max. `sweep_int8.py` screens alternative
policies approximately before engine compilation. `test_quantize_int8.py`
checks small-graph Q/DQ construction, per-channel axes, smoothing error bounds,
and output protection. `benchmark.py --capture-root DIR` records full
trajectories alongside the usual device-wide memory and RSS measurements.

## Decision

Keep this as an opt-in experiment. More aggressive INT8 saved substantial SS
weight storage but failed numerical checks; the mixed policy passes those
checks and produces assets, yet full-trajectory quality still fails. A useful
next accuracy experiment would examine accumulated error across chest sampling
steps, identify sensitive layers/timesteps, and validate a revised policy on
additional images and seeds. Quantization-aware training is a larger follow-up
if calibrated mixed precision cannot meet quality requirements.

For larger full-pipeline memory savings, shape and texture flows now deserve
profiling. SS-only weight compression does not remove their allocations.
The [CUDA memory follow-up](low-vram-cuda.md) adds FP16/BF16 weight streaming
and bounded decoder working sets without requiring INT8.
Actual in-process Glint generation, upload/build memory, viewport contention,
and visual acceptance remain open. This experiment does not close those gates.
