# Native worker API and TensorRT SS flow

The first native milestone runs the full fixed-512 image-to-textured-GLB pipeline
on Windows/CUDA. TensorRT optionally replaces only the sparse-structure (SS) flow
forward pass. The native sampler, shape/texture flows, decoders, mesh processing,
and PBR baking remain in C++/GGML. Python is used offline for export and analysis.

The tested engine uses **BF16**, with FP32 inputs and outputs. The FP16 probe
produced nonfinite predictions and is not an accepted runtime configuration.
An [INT8 follow-up experiment](int8-tensorrt.md) runs with mixed precision but
fails full-trajectory quality gates; BF16 remains the reference configuration.
FP16 and BF16 both store two bytes per scalar; switching between them does not
halve weight storage.

This report validates the standalone milestone on the user's **RTX 5080**.
The Glint UI/MCP worker path and undoable scene append remain a later handoff.
The editor's complete generation/upload/acceleration-build memory budget has
not been validated. A material-side silhouette also needs quality review before
making TensorRT the default backend.

## Pinned environment

| Component | Tested revision/version |
| --- | --- |
| trellis.cpp baseline | `6851891d706e2e26a8bb9dd99f28de851d68b64d` |
| Branch | `onnx-tensorrt` |
| GGML submodule | `f8697b33795db2ee6a2569b40853ca6aaa69f9b9` |
| Microsoft TRELLIS.2 source | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` |
| Microsoft model repository | `af44b45f2e35a493886929c6d786e563ec68364d` |
| GGUF model repository | `a57397bd3d351599d9729fc144b3f87c3f87d65b` |
| TensorRT headers | NVIDIA/TensorRT `72fc1a3d23a163216893b363fb562b9464865afb` |
| TensorRT runtime | `11.4.0.106`, CUDA 13 package |
| GPU / driver | RTX 5080 16 GiB / 617.14 |
| CUDA toolkit | 13.3.73 |
| Compiler | VS 2022 MSVC 19.44.35226, x64 |
| Offline Python / PyTorch | 3.11.0 / 2.11.0+cu128 |

`tools/onnx/fetch_models.py` pins downloads and checks LFS SHA-256 hashes. Its
`models/manifest.json` records every file. The original SS checkpoint SHA-256 is
`ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6`;
the native SS GGUF is
`1dfbef1b80ddea422a6189ff9ac29a244c4482cfbaa01870a0c0ea76d3fb9ae6`.
Weights, exported graphs, engines, and raw captures are ignored build artifacts.

The tested full engine is 2,632,776,252 bytes, SHA-256
`9b2afd4892a270878b5fb5042e043742e2427782104645156478b84fd3fdc0bd`.
Rebuild engines for the intended hardware/runtime; a serialized plan is not a
portable replacement for its ONNX graph and checkpoint provenance.

## Build and run

Install VS 2022 C++ tools, CMake, Ninja, CUDA, and matching TensorRT headers/runtime.
The headers must expose `include/NvInfer.h` under `TensorRTRoot`. Runtime DLLs must
be on `PATH`; linking an import library is unnecessary because the optional
backend loads the TensorRT runtime dynamically.

```powershell
git submodule update --init --recursive
.\scripts\build-onnx-windows.ps1 -BuildDir build-onnx -TensorRTRoot out/TensorRT -Test

$env:PATH = "$PWD\.venv\Lib\site-packages\tensorrt_libs;$env:CUDA_PATH\bin\x64;$env:PATH"
.\build-onnx\trellis-job.exe assets/showcase/axe/axe.png out/axe.glb models/f16 `
  --ss-engine out/onnx-tensorrt/ss_flow.plan
.\build-onnx\trellis-job.exe assets/showcase/axe/axe.png out/lifecycle.glb models/f16 `
  --ss-engine out/onnx-tensorrt/ss_flow.plan --exercise-lifecycle
```

Omit `--ss-engine` for GGML SS flow. The CLI and server still work; the CLI also
accepts `--ss-engine FILE` and `--capture-dir DIR`. An engine error fails the job
instead of silently selecting GGML. TensorRT can be disabled with
`-DTRELLIS_TENSORRT=OFF` (the default). A CPU build still compiles and runs the
contract tests; the generation job API deliberately requires CUDA.

Local verification used the pre-existing `build-baseline` directory for the final
modified build. The pristine baseline binaries were copied to `out/baseline-bin`
before changes; the directory name alone must not be used to identify a baseline.

### Offline export

Create a Python 3.11 environment and install a CUDA-enabled PyTorch separately,
then install `tools/onnx/requirements.txt`. The development environment inherited
the already-installed PyTorch via `--system-site-packages`; this is optional.
Check out the Microsoft source and TensorRT headers at the revisions above.

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/onnx/fetch_models.py
.\.venv\Scripts\python.exe -X utf8 tools/onnx/test_export.py --reference out/TRELLIS.2
.\.venv\Scripts\python.exe -X utf8 tools/onnx/export_ss_flow.py `
  --reference out/TRELLIS.2 `
  --checkpoint models/reference/ckpts/ss_flow_img_dit_1_3B_64_bf16 `
  --output out/onnx-tensorrt/ss_flow.onnx --precision bf16
.\.venv\Scripts\python.exe -X utf8 tools/onnx/build_engine.py `
  out/onnx-tensorrt/ss_flow.onnx out/onnx-tensorrt/ss_flow.plan
.\.venv\Scripts\python.exe -X utf8 tools/onnx/inspect_attention.py `
  out/onnx-tensorrt/ss_flow.plan
```

The exporter accepts `--fixtures DIR` for captured native inputs and
`--block-only` for a cheap feasibility probe. `validate.py ONNX FIXTURES` runs ONNX
Runtime CPU as a debugging check. The actual ORT execution check used a one-block
FP32 model: relative L2 error `6.50e-7`, maximum absolute error `1.335e-5`.
Full BF16 numerical validation used TensorRT and the original PyTorch model;
do not interpret the ORT result as full BF16 graph validation.

TensorRT 11 uses strongly typed graphs. BF16 is encoded by model dtypes/casts
rather than a builder-wide precision override. The exporter replaces complex
RoPE arithmetic with an equivalent real-valued form and compares the result
against the untouched model before export. The independent RoPE operator test
uses a tight `1e-6` tolerance. Whole-network BF16 adaptation has a looser 0.02
relative/absolute tolerance because rounding accumulates across blocks.

## Library contract

`include/trellis_job.h` exposes synchronous `trellis::run_job`. The caller supplies
encoded image bytes, an explicit seed (including zero), and the 512 preset.
The default is 12 steps, 150,000 target faces and a 1024 atlas. Successful jobs
return PBR GLB bytes in memory, timings, actual stage backend/precision, sampled
GPU memory, guidance-clamp counts, and nonfinite-repair counts. Failure and
cancellation return no asset.

Call it from a dedicated worker thread. Progress callbacks execute on that thread;
marshal UI updates to the editor thread. Do not recursively invoke a job from a
callback. Jobs are serialized because upstream runtime flags are process-wide.
Cancellation is cooperative between forwards and major stages. Long kernels can
finish before cancellation returns; pending GPU work is drained before release.
Models and engine buffers have explicit RAII lifetimes. Weights load one stage
at a time and engine buffers are reused between denoising steps.

The SS interface is token-major, channel-contiguous FP32:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `latent` | `[1,4096,8]` | Noisy 16-cubed latent grid |
| `timestep` | `[1]` | Native timestep already multiplied by 1000 |
| `cond` | `[1,1029,1024]` | Image or negative conditioning |
| `velocity` | `[1,4096,8]` | Model prediction |

RNG, scheduling, CFG order/rescaling, and Euler updates stay in the native
sampler. The current boundary copies host inputs/outputs on a dedicated CUDA
stream and synchronizes each forward. Those transfers are included in timings;
this is not a zero-copy interop implementation. Captures include every SS forward,
the actual initial noise for all flows, final SS latent, and decoded SS logits.

## Measured results

The [machine-readable report](benchmarks/onnx-tensorrt-5080/summary.json) contains
input hashes, provenance, engine metadata, numerical checks, and limitations.
Raw local logs and assets are under `out/onnx-tensorrt`.

Both paths used seed 42, 512 resolution, 12 steps, 150,000 target faces, the same
automatic background removal, 1K PBR atlas, and PNG textures. Each fixture had
two process launches. The second benefits from OS/driver caches, but models still
reload: these are not persistent-engine warm timings. The saved Glint scene and
desktop remained resident. Some development builds ran during measurement;
these numbers describe this workstation session rather than an isolated lab.

| Fixture | GGML mean seconds | BF16 TensorRT mean seconds | Speedup | TRT additional peak GiB |
| --- | ---: | ---: | ---: | ---: |
| Chest / opaque prop | 78.52 | 69.16 | 1.14x | 3.55 |
| Axe / thin geometry | 26.80 | 22.96 | 1.17x | 3.21 |
| Drone / disconnected parts | 50.90 | 46.14 | 1.10x | 3.29 |
| Knight / varied material | 31.57 | 28.05 | 1.13x | 3.22 |

Total device usage reached approximately **7.46–7.80 GiB** with TensorRT,
including about **4.24 GiB already resident**. GGML reached about 7.28–7.78 GiB,
adding 3.04–3.54 GiB. Baseline process RAM peaked around 3.03 GiB; TensorRT around
3.21 GiB. TensorRT improves time modestly but does not reduce peak memory here.
The GGML path stores FP16 weights and uses BF16 attention; neither measurement
should be described as a pure FP16 TensorRT pipeline.

Device-wide memory was sampled with `nvidia-smi` every 50 ms, so observed peaks
are lower bounds. Per-stage library counters use `cudaMemGetInfo` every 20 ms;
WDDM accounting differs from the device-wide readings. Do not subtract the two
different accounting systems to derive a memory budget.

The chest SS sampling loop (22 conditional/unconditional forwards) fell from
roughly 9.2 to 3.8 seconds, excluding weight load. The final worker verification
axe job took 23.75 seconds, including 5.76 seconds in SS loading/execution.
Its highest sampled stage usage was SS flow; flow weights dominate this peak.
The engine's context scratch is 307,697,664 bytes (293.4 MiB). Inspection finds
60 fused attention operations, and total scratch is below even one full BF16
`[1,12,4096,4096]` score tensor (384 MiB). This evidence is scoped to this fixed
graph, standard TensorRT operators, and the tested GPU.

### Numerical and quality limits

Early, negative-conditioning, middle, and late TensorRT forwards against the
original PyTorch model had relative L2 errors 0.51%, 0.61%, 0.63%, and 1.09%.
Maximum absolute errors were 0.0314, 0.0326, 0.0847, and 0.1621. The provisional
gate is **L2 <= 2% and maximum error <= 0.02 + 5% of reference signal range**.
This was widened from the initial pointwise 2% test; late values do not all pass
pointwise `allclose` (482/32768 exceed it). Diagnostics retain that distinction.
These tolerances require product-level quality signoff, not just a passing test.

Native versus TensorRT final SS decoded occupancy IoU was **98.75%**. The native
sampler retains upstream guidance-rescaling clamps; Microsoft's sampler does
not. With identical actual initial noise, the raw Microsoft/native trajectory
therefore differs by 11.38% relative latent L2, despite 99.3% decoded occupancy
IoU. This is not exact sampler parity. Successful BF16 runs needed no nonfinite
repairs and recorded six guidance clamps.

All 16 generated GLBs passed structural/PBR checks. Minimum silhouette IoU over
four fixed orthographic views was chest 98.29%, axe 96.45%, drone 95.89%, and
knight **94.59%**. The knight's side view falls below the provisional 95% gate;
15/16 views pass. Baseline repeated-run minimum IoU was 99.37%, so that difference
is larger than repeat variation. Keep TensorRT optional pending visual review.
Matching full textured Microsoft reference assets have not been generated;
reference comparisons currently cover dense SS predictions and occupancy.

### Repeated use and Glint evidence

Both GGML and TensorRT passed success -> cancellation after SS step one ->
missing-engine failure -> success in one process, with valid successful assets.
Warmed idle `cudaMemGetInfo` usage stabilized at 1,571,225,600 bytes for GGML and
1,596,391,424 bytes for TensorRT. A subsequent two-job TensorRT run returned to
the same idle value. Contract tests cover invalid bytes, pre-cancellation,
mid-sampler cancellation, image preprocessing, ownership, GLB serialization,
and output failure.

`tools/glint_import` links the existing local Glint IO library and verifies
in-memory import. All four TensorRT assets retained one mesh, one material,
and both embedded PBR images. A fixed-camera, lit axe asset was rendered by the
existing Glint CUDA/OptiX benchmark executable. A second render ran concurrently
with generation: frame wall-time p95 rose from 9.90 ms isolated to 31.60 ms,
with a maximum of 53.50 ms. These are separate-process headless measurements,
not editor UI responsiveness or in-process worker validation. The Glint binary
was prebuilt locally; its source revision is not pinned by this experiment.

## Next decision

Current [resident profiles](resident-profiles.md) use BF16 by default and
experimental INT8 optimization, both without weight streaming. The separate
[CUDA memory experiment](low-vram-cuda.md) records historical weight streaming
and the smaller decoder working sets retained by the resident profiles.

The standalone four-fixture workload is below the initial 6 GiB additional
target on the 5080. Stage-by-stage loading and fused attention already avoid
the larger obvious allocations. FP8 remains deferred. Before a second ONNX
port, profile representative assets in Glint: mesh/bake and sparse shape work
are now significant residual costs, and the current tests do not justify a
specific second export yet.

The remaining Glint handoff must share one native job path between UI and MCP,
append the returned GLB as one undoable edit, and measure generation together
with texture upload, acceleration-structure build, editor frame times, and total
device memory. Keep the current Glint scene and unrelated working-tree edits
intact. The full editor budget and final visual acceptance remain open.
