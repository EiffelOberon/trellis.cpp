# Apple silicon worker and MLX SS flow

The native 512 worker accepts `--device metal` and uses GGML by default.
`--ss-backend mlx --ss-weights FILE` replaces the dense sparse-structure
forward. Both paths use the native noise, schedule, guidance, Euler updates,
sparse flows and decoders, mesh processing, and PBR bake. The worker returns
embedded PNG textures through its in-memory GLB API.

## Reproduce the build

The development Mac was a MacBook Pro Mac15,8 with an Apple M3 Max, a 40-core
Apple9 GPU and 64 GiB unified memory. It ran macOS 27.2 (26B123), Xcode 27.2
(27B52), SDK 27.2, Apple Clang 21.0.0 (2100.4.1.1), and CMake 4.4.5. The
Trellis base is `45edf32c2c5031932b735a9db38accba8e03d704`, and pinned GGML is
`f8697b33795db2ee6a2569b40853ca6aaa69f9b9`. The local Glint checkout is
`3e1c766a4e2bbdd51f2fa6ab33589b7c4dd1e488`.

```sh
git submodule update --init --recursive
mkdir -p out
git clone --branch v0.32.4 https://github.com/ml-explore/mlx.git out/mlx
git -C out/mlx checkout 36969c094354086e15d2a327a9c50feb20b9cf11

cmake -S out/mlx -B build-mlx-sdk-shared \
  -DCMAKE_INSTALL_PREFIX="$PWD/out/mlx-sdk-shared" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=27.2 \
  -DBUILD_SHARED_LIBS=ON -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_GGUF=OFF
cmake --build build-mlx-sdk-shared --parallel 8
cmake --install build-mlx-sdk-shared

cmake -S . -B build-mlx-pinned \
  -DMLX_DIR="$PWD/out/mlx-sdk-shared/share/cmake/MLX" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=27.2 -DTRELLIS_MLX=ON \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DTRELLIS_WEBP=OFF
cmake --build build-mlx-pinned --parallel 8 \
  --target trellis-job trellis-cli trellis-probe-mlx trellis-test-job trellis-compare-ss
ctest --test-dir build-mlx-pinned --output-on-failure
```

Use `-DTRELLIS_MLX=OFF` for the GGML-only build, and `-DGGML_METAL=OFF`
as well for the CPU contract build. The core stays on C++17. The MLX target and
probe use C++20. The tested package targets macOS 27.2. Qualification on older
macOS versions and other GPU families is open. [MLX C++ build documentation](https://ml-explore.github.io/mlx/build/html/dev/mlx_in_cpp.html)

## Weights and fixtures

Run `tools/onnx/fetch_models.py` with `huggingface-hub`. It pins the GGUF and
Microsoft Hub revisions from the [TensorRT report](onnx-tensorrt.md), checks each
downloaded LFS file, and writes `models/manifest.json`. The MLX runner consumes
the original Microsoft safetensors directly. The fixed architecture comes from
the pinned config below. The loader verifies each tensor name, shape and BF16
storage type and rejects extra tensors. It transposes the original
`[out_channels, in_channels]` linear weights for MLX matmul and casts at load
time. The MLX worker uses FP16 for the transformer, with FP32 input, time and
output projections, normalization, RoPE and GELU calculation. It converts and
evaluates checkpoint tensors one block at a time to bound temporary weight
allocations. The probe also supports BF16 and FP32. FP32 diagnostic runs set
`MLX_ENABLE_TF32=0`.
[MLX precision documentation](https://ml-explore.github.io/mlx/build/html/usage/precision.html)

| Input | SHA-256 |
| --- | --- |
| SS safetensors | `ca01377c485bec418076d38ee80166d32dc776d744f2553b835cba1e97a7abf6` |
| SS config | `6cdfb636854f60ffcadd8c49244169cd449f34c5ccdb55b5ff7c04678f9c399e` |
| Axe PNG | `f7f21b63607c5ba9a0fce954273351b2ddbde26aa62bb5d565d69a4a2ebc88a2` |
| Chest PNG | `a33fe731d46bc34d8ade14fded885a1bf0c152e2792fbe8c10bf561c938420e7` |
| Drone PNG | `b09f3797ec8a61dff00d9b4d4026474727940a826172571784869e732b0c65bd` |
| Knight PNG | `05315f8c5f8dc1d14e702ffa59846531b5b8d6e1b12709fd945cc14ff1d28192` |

```sh
mkdir -p out/sample
weights=models/reference/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors
build-mlx-pinned/trellis-job assets/showcase/axe/axe.png out/sample/axe.glb models/f16 \
  --device metal --ss-backend ggml --capture-dir out/sample/capture
MLX_ENABLE_TF32=0 build-mlx-pinned/trellis-job assets/showcase/axe/axe.png \
  out/sample/axe-mlx.glb models/f16 --device metal --ss-backend mlx \
  --ss-weights "$weights" --capture-dir out/sample/capture-mlx
```

The harness uses seed 42, 12 steps, a 150,000-face target, automatic matte
selection, and a 1024 PBR atlas. `--capture-dir` stores every SS input/output,
the actual initial noise for all flows, the final SS latent, and decoded logits.
The full reference script selects early, negative, middle, and late inputs
and writes their hashes, the prediction hashes, tool versions, and comparison
metrics. Use `--block` for the intermediate outputs, `--precision fp32|fp16`
for diagnostic dtypes, or `--native-latent` to compare Microsoft's sampler
separately from the native guidance clamps.

```sh
git clone https://github.com/microsoft/TRELLIS.2.git out/TRELLIS.2
git -C out/TRELLIS.2 checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
MLX_ENABLE_TF32=0 python tools/onnx/mlx_reference.py \
  --reference out/TRELLIS.2 --checkpoint "$weights" --capture out/sample/capture \
  --probe build-mlx-pinned/trellis-probe-mlx --precision fp16 --output out/sample/reference
```

Offline reference verification used Python 3.12, PyTorch 2.15.0, NumPy, and
safetensors. The original-sampler probe also uses easydict and tqdm.
`tools/onnx/benchmark_metal.py` takes `--exe`, `--output`, and
optional `--weights`, and uses Pillow and NumPy for GLB and silhouette checks.
It runs one fresh process and then three timed jobs after a warm-up in a second
process, per fixture and backend. Each job reloads weights. Keep the background
work and render settings fixed between backend measurements. Set
`TRELLIS_DBG_ALLOC=1` to log direct GGML graph buffer sizes and
`TRELLIS_DBG_PLACEMENT=1` to inventory scheduler assignments when enabled.

The primary Mac memory metric is the process physical footprint
(`TASK_VM_INFO.phys_footprint`), sampled every 20 ms through GLB serialization.
The additional footprint is the sampled peak minus the idle sample at the
start of that job. MLX active, cache, and peak allocator readings are logged
separately. The CUDA counters appear as JSON `null` for Metal jobs. Benchmark
system snapshots record swap, VM statistics, and memory pressure before and
after each group. CPU work includes the preprocessing and deformable-convolution
host paths, sparse index construction and mesh extraction, remeshing, QEM,
bake, and serialization. The explicit device indicates where the GGML models
execute.

## Attention and package smoke tests

Both attention shapes use `force_fused=true` so MLX raises an error if
a fused kernel is unavailable. Evaluate and synchronize before collecting
peak usage or reading output.

```sh
MTL_CAPTURE_ENABLED=1 MLX_ENABLE_TF32=0 build-mlx-pinned/trellis-probe-mlx \
  attention out/self.gputrace 4096 bf16
MTL_CAPTURE_ENABLED=1 MLX_ENABLE_TF32=0 build-mlx-pinned/trellis-probe-mlx \
  attention out/cross.gputrace 1029 bf16
```

On the M3 Max both BF16 traces were produced. Each isolated operation added
12 MiB to active allocator peak after evaluating the inputs, equal to the
output buffer. The dense scores alone would be 384 MiB for self-attention and
96.47 MiB for cross-attention. The isolated FP16 and FP32 probes also passed,
with 12 MiB and 24 MiB additional peak respectively. These allocator readings
exclude per-process Metal and driver overhead.

For this shared build, colocate the worker with the GGML `libggml*.0.dylib`
files, `libmlx.dylib`, `libjaccl.dylib`, and `mlx.metallib`. GGML's Metal
sources are embedded by the build option. Give the executable an
`@loader_path` rpath in place of its absolute build and SDK rpaths, then
sign the packaged executable. The GGML and MLX licenses are MIT. Include their
license texts and the SDK's dependency notices, including JACCL, fmt,
nlohmann/json, and Metal C++ when redistributing the build. Model checkpoint
distribution follows the model repository's terms. Pass an explicit model
directory and SS checkpoint path at runtime.

```sh
bundle="$PWD/out/mlx-package"
sdk="$PWD/out/mlx-sdk-shared"
build="$PWD/build-mlx-pinned"
mkdir -p "$bundle"
cp "$build/trellis-job" "$bundle"
cp -L "$build"/libggml{,-base,-cpu,-blas,-metal}.0.dylib "$bundle"
cp "$sdk"/lib/{libmlx.dylib,libjaccl.dylib,mlx.metallib} "$bundle"
install_name_tool -rpath "$sdk/lib" @loader_path "$bundle/trellis-job"
install_name_tool -delete_rpath "$build" "$bundle/trellis-job"
codesign --force --sign - "$bundle/trellis-job"
```

The packaged smoke test runs a relocated executable with
`env -i PATH=/usr/bin:/bin` and a single `@loader_path` rpath. `--exercise-lifecycle` runs success,
cancellation after SS step one, missing weights, and recovery in one process.
Combine it with `--repeat 11` to append ten further completed jobs in the same
process. A failed or cancelled job must publish no GLB.

The extended residency experiment uses one fixed alpha cutout of the axe
to keep the repeated workload tractable. Generate it once using the same
automatic background removal and retain its hash.

```sh
mkdir -p out/mlx-lifetime
build-mlx-pinned/trellis-cli assets/showcase/axe/axe.png out/mlx-lifetime/axe.glb \
  --models models/f16 --device metal --res 512 --bg-only
shasum -a 256 out/mlx-lifetime/axe_cutout.png
env -i PATH=/usr/bin:/bin MLX_ENABLE_TF32=0 out/mlx-package/trellis-job \
  out/mlx-lifetime/axe_cutout.png out/mlx-lifetime/axe.glb models/f16 \
  --device metal --ss-backend mlx --ss-weights "$weights" --exercise-lifecycle --repeat 11
```

## Glint handoff

The optional integration on Glint's `codex/trellis-metal` branch links the
native worker and copies its Metal resources beside the editor. Configure
Glint with the following CMake cache entries.

```sh
-DGLINT_TRELLIS_SOURCE_DIR=/path/to/trellis.cpp
-DGLINT_TRELLIS_BUILD_DIR=/path/to/trellis.cpp/build-mlx-pinned
-DGLINT_TRELLIS_PACKAGE_DIR=/path/to/trellis.cpp/out/mlx-package
```

Set `GLINT_TRELLIS_MODELS` to the GGUF directory before launching Glint.
Set `GLINT_TRELLIS_SS_WEIGHTS` to the original safetensors to enable the MLX
choice. File > Generate 3D from Image and the MCP tool
`glint_generate_3d_asset` use the same worker service. MCP uses Glint's existing
approved input directory, document revision checks, and request progress and
cancellation. The worker decodes generated GLB bytes, and the owner thread
publishes a single undoable append into the current scene generation. Replacing
the scene supersedes the job.

Glint's `editor/tests/mcp_trellis_integration.py` exercises the dedicated editor
scene, MCP generation, an intervening document edit, cancellation, and undo/redo.
Use `--visible` for a windowed run. The test writes CPU and GPU frame time
logs and the editor logs the append transaction time separately. For interactive
qualification at 640×360, use a 33.3 ms p95 frame budget and investigate
individual stalls above 100 ms. Record the renderer settings and comparable
background load for both SS backends.

## M3 Max development results

The [machine-readable report](benchmarks/mlx-metal-m3-max/summary.json) records
the verified downloads, fixture hashes, forward errors, stage timings, and
observed memory. Background builds, reference replays, and editor activity
overlapped the timed jobs at different points. These are development
measurements. The controlled repetition matrix remains open.

### Numerical checks

The FP32 one-block results stayed within `8.9e-7` relative L2 at the four
tapped outputs. BF16 and FP16 intermediate checks passed. The complete FP16
late-forward probe also produced finite outputs. For BF16 on captured axe
inputs, the four full forwards against the original PyTorch MPS model measured
0.227%, 0.146%, 0.196%, and 0.600% relative L2. Maximum absolute error reached
0.125 on the late forward, and 132 of 32,768 values there exceeded pointwise
allclose. All four passed the provisional relative-L2 and signal-relative
maximum-error gates.

A sequential SS-only comparison on the captured early axe input measured
first forwards at 5.105 s for GGML Metal and 1.550 s for BF16 MLX. The
three warm forwards had medians of 5.094 s and 1.549 s respectively,
a 3.29× speedup for that forward. GGML setup included initial Metal
library compilation. End-to-end timing includes the other pipeline stages
and still needs the controlled repetitions.

The separate Microsoft sampler replay used the native axe noise and
conditioning with Microsoft's original guidance policy. Its final latent
differed from native GGML by 11.27% relative L2. Decoding both on the same
native Metal decoder gave 99.15% occupancy IoU.

An isolated late-forward submission experiment evaluated one- and five-block
groups. Both produced byte-identical velocity arrays. The one-block variant
measured 3.85 GB peak MLX active allocations against roughly 4.19 GB in the
full-forward probe. Controlled viewport and end-to-end measurements remain
open for the grouped prototype.

### Textured fixtures

All eight GLBs passed structural checks and Glint in-memory import with two
embedded 1024 PNG textures and their PBR bindings. Footprint is the increase
from the job's idle sample to its sampled peak.

| Fixture | GGML seconds | BF16 MLX seconds | GGML GiB | MLX GiB | Occupancy IoU | Minimum silhouette IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Axe | 274.5 | 331.7 | 6.51 | 7.99 | 99.50% | 100.00% |
| Chest | 787.8 | 701.0 | 6.50 | 7.89 | 98.96% | 97.79% |
| Drone | 491.7 | 324.3 | 6.50 | 7.90 | 96.07% | 96.36% |
| Knight | 566.2 | 355.7 | 6.50 | 7.90 | 98.30% | **94.06%** |

Glint's headless Metal renderer produced matched 640×360, 32-sample
fixed-light images under `out/glint-qa`. The axe PNGs were byte-identical.
The GGML chest shows a clearer right-side ring and brighter metal band
details. Both drone renders preserve the shell and separated details,
with variation around the lens and underside. The knight missed the 95%
silhouette criterion at 90° (94.06%) and 300° (94.62%). Armor and cape
shape and highlights vary in the fixed-light images.

### Memory and lifetime

The relocated worker completed the axe, cancelled after SS step one, reported
a missing-weight failure, and completed a second valid axe in one process.
A malformed checkpoint also failed cleanly in the packaged probe. MLX
reported zero active and cached allocator bytes after SS release. Process
footprint was about 6.0 GiB immediately after cancellation and 172 MiB after
recovery. Both backends' first-process fixture peaks exceeded the experimental
6 GiB additional-footprint target.

The extended alpha-cutout test completed the same recovery sequence plus
ten further successful jobs in one process. All twelve produced GLBs passed
PBR structure checks. The ten post-recovery physical-footprint readings
ranged from 402 to 958 MiB with a median of 842 MiB and no monotonic growth.
MLX reported zero active and cached allocator bytes after all thirteen SS
releases in that process.

### Editor verification

The Glint MCP integration passed generation, an intervening edit, one-step
undo/redo, and cancellation in a dedicated scene with the Metal viewport
rendering. The editor also exited normally while an asynchronous generation
job was active, before asset publication. The MLX run recorded 43.1 ms for
append, including an 11.7 ms BLAS build and a 0.5 ms TLAS build.

An independent GGML job also used the GPU during that MLX editor run.
Viewport CPU wall p95 was 37.9 ms and GPU p95 was 46.2 ms during generation.
The largest observed frame took 5.0 s in host presentation retention near
the transition to shape flow. Controlled viewport qualification and the
interactive File menu check are open.

GGML is the worker default. MLX remains an opt-in backend for evaluation
against the outstanding quality, footprint, and viewport criteria.

A same-input knight diagnostic compared BF16, FP16, and FP32 against GGML
for four saved forwards. Late relative L2 to GGML was 1.326% with BF16,
0.427% with FP16, and 0.397% with FP32. A scratch end-to-end FP16 build
passed the knight gates with 99.15% occupancy IoU and a 95.88% minimum
silhouette IoU. Those results motivated using FP16 for MLX worker jobs.

### FP16 quality comparison

All four FP16 assets met the 95% decoded-occupancy and four-view silhouette
gates against the GGML Metal fixtures. Glint imported each GLB with both
embedded PBR textures and their material bindings. The
[FP16 results](benchmarks/mlx-metal-m3-max/fp16-quality.json) record each
view, timing, allocation, reference error and fixed-light render hash.

| Fixture | BF16 minimum silhouette IoU | FP16 minimum silhouette IoU | FP16 occupancy IoU |
| --- | ---: | ---: | ---: |
| Axe | 100.00% | 100.00% | 99.75% |
| Chest | 97.79% | 98.87% | 98.90% |
| Drone | 96.36% | 97.64% | 98.80% |
| Knight | 94.06% | 95.88% | 99.15% |

The rebuilt worker reproduced all 22 captured velocity arrays, the final
latent, decoded logits, and mesh geometry from the FP16 knight diagnostic.
The matching Glint render also produced the same PNG. Against the GGML
Metal knight, silhouette IoUs at 0°, 90°, 180°, and 300° were 98.24%,
95.88%, 98.24%, and 96.33%.

The FP16 axe and GGML axe produced identical fixed-light PNGs. In the other
matched Glint renders, the chest side ring and metal bands, drone underside
details, and knight sword and cape remain visible. Differences include chest
trim and stud highlights, the drone underside, and knight shoulder highlights.

On the same four captured GGML knight inputs, FP16 comparisons against
Microsoft's FP16 reference on PyTorch MPS passed both forward gates and
pointwise allclose for every value. Relative L2 was 0.030%, 0.020%, 0.029%,
and 0.108% for the early, negative, middle, and late forwards respectively.

On the captured early knight input, median warm forward time across three
sequentially timed forwards was 1.649 seconds for FP16 MLX and 5.099 seconds
for GGML Metal, a 3.09× speedup for that stage. Across the four full FP16
jobs, the additional process footprint was 7.74–7.81 GiB, above the 6 GiB
experimental target. Controlled end-to-end and integrated viewport
measurements remain open, and GGML remains the default.

### GGML editor handoff

The same Glint MCP integration also passed with GGML in an editor
executable using a single `@loader_path` rpath. Append took 40.2 ms.
With simultaneous standalone tests, the wall-time log recorded an 11.95 s
stall in the renderer call. GPU counters repeatedly held multi-second
values across subsequent frames and the pre-worker baseline had just three
qualifying frames. These measurements leave a controlled comparison open.
CUDA hardware regression runs remain open.
