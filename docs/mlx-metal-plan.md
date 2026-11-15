# TRELLIS.2 MLX / Metal plan for Glint

The worker implementation and qualification evidence are tracked in the
[Apple silicon build and validation guide](mlx-metal.md).
Starting point: trellis.cpp `456a000b98ec18eaf8805e357f2684f465af1c7e`.
Create `codex/mlx-metal` when implementation starts.

## Outcome and scope

Run the complete 512-cubed image-to-textured-GLB job on Apple silicon through
the same native worker API used on Windows. First establish GGML Metal as the
Mac baseline, then add an optional **MLX C++ implementation of the dense
sparse-structure (SS) flow forward pass**. Keep RNG, scheduling, guidance, Euler
updates, shape/texture flow, sparse decoding, meshing, and baking in the existing
native pipeline. Return an in-memory GLB with a 1K PBR atlas.

Glint UI and MCP must share that job path. Production acceptance includes an
undoable asset append and measurements with the Metal viewport rendering during
generation, texture upload, and acceleration-structure construction.

MLX provides a C++ library usable through CMake and a Metal backend. It is a
candidate execution framework for this port; speed and memory benefits remain
unmeasured. Implement the model using MLX operations and checkpoint weights;
the TensorRT engine is not the Mac runtime artifact.
[MLX C++ integration](https://ml-explore.github.io/mlx/build/html/dev/mlx_in_cpp.html)

Python is allowed for offline weight conversion, reference comparisons, and
benchmark analysis. The shipped worker must run without a Python interpreter.
Shape/texture MLX ports, quantization, higher resolutions, custom sparse Metal
kernels, and direct renderer-buffer interop are conditional follow-up work.

## 1. Pin a Mac environment and make fixtures reproducible

- Select an Apple silicon Mac for development; record chip/GPU family, unified
  memory capacity, macOS build, Xcode/SDK, compiler, CMake, MLX revision, GGML
  revision, Glint revision, and build flags. No Mac target has been supplied yet.
- Pin a tested MLX release/source commit rather than following `main`. Record
  licenses and the required runtime resources for redistribution. Verify the
  minimum supported macOS and GPU family against the actual dependency build.
- Start from the Microsoft/GGUF revisions and SHA-256 manifests in
  [the TensorRT report](onnx-tensorrt.md). Package the captured early, negative,
  middle, and late SS forwards, actual initial noise, final latents, and reference
  predictions with hashes. They currently live in ignored `out` directories;
  document a reproducible capture/download procedure instead of relying on one
  developer's filesystem.
- Reuse chest, axe, drone, and knight inputs with seed 42, 512 resolution,
  12 steps, 150,000 target faces, PNG textures, and a 1024 PBR atlas. Preserve
  preprocessing settings and input hashes. For isolated forward comparisons,
  use the captured conditioning bytes to remove image-encoder variation.
- Validate on one named Mac first. Define the supported hardware matrix only
  after testing; require another GPU generation if claiming cross-generation
  support. The Windows RTX 5080 results are context, not Mac acceptance evidence.

**Exit:** a pinned build environment and independently verifiable fixture bundle.

## 2. Establish the complete GGML Metal baseline

The existing CMake file links `ggml-metal` on Apple builds. However,
`src/trellis_job.cpp` explicitly sets `params.backend = "CUDA"`, while its memory
monitor and failure cleanup use CUDA APIs. Build support alone does not make the
new worker API usable on Metal.

- Add explicit execution-device selection to `JobRequest`, separately from the
  SS model implementation. Preserve existing Windows/CUDA behavior. Expose the
  same selection in the worker harness and CLI; reject unsupported combinations
  such as TensorRT on Metal before allocating models. Report the resolved device.
- Put synchronization, device capability queries, and memory sampling behind
  backend-specific implementations. Use supported GGML Metal synchronization
  rather than a CUDA-only drain. Represent unavailable counters explicitly;
  a zero value must not masquerade as measured zero GPU use.
- Run all stages with GGML Metal. Inventory CPU execution, unsupported operations,
  and copies. CMake currently documents CPU fallbacks for deformable convolution
  and QEM decimation; inspect the rest of the graph too. Label intentional CPU
  work and reject an unavailable requested GPU. Do not hide CPU stages in an
  overall label claiming that everything ran on Metal.
- Measure conditioning, all flows, each decoder, mesh processing, bake, and
  serialization. Separate first process launch, first evaluated model call, and
  subsequent calls/jobs; include any shader compilation and weight loading.
- Generate the four textured assets and perform GLB/PBR, occupancy, silhouette,
  and fixed-lighting render checks. Retain the current native sampler's guidance
  clamps; compare the unmodified Microsoft sampler separately because its policy
  differs. Avoid changing sampler behavior during a backend comparison.

**Exit:** four valid GLBs from the worker API on a Mac, with actual stage backend,
precision, timing, and memory evidence. Resolve baseline correctness failures
before attributing differences to MLX.

## 3. Prove one MLX block and select precision

- Add a narrow optional `trellis_mlx` target and `TRELLIS_MLX` build option,
  disabled by default and guarded for the tested Apple configuration. Keep MLX
  headers out of public job headers with a pimpl runner. Apply dependency C++
  standard requirements to the MLX target; the current core uses C++17, while
  MLX's documented example uses C++20. Verify a complete executable link early.
- Load the pinned original SS safetensors/config from C++, or produce a
  deterministic offline conversion with an explicit tensor-name/layout mapping.
  Fail on missing, unexpected, wrong-shape, or wrong-dtype tensors. Record source
  and converted file hashes, dtype changes, and conversion-tool versions.
- Implement one block with real checkpoint weights: projections, normalization,
  time modulation, RoPE, self-attention, cross-attention, and the feed-forward
  path. Compare intermediate activations against the original PyTorch model.
- Start with an FP32 diagnostic case and verify whether the pinned MLX backend
  uses reduced internal precision. Record precision-related environment flags.
  Test BF16 and FP16 on the selected Mac rather than assuming NVIDIA behavior
  transfers to Apple silicon. Use higher precision for sensitive operations
  where required; treat nonfinite predictions as a hard error.

MLX documents that FP32 matrix operations can use reduced internal precision
depending on hardware/backend. Explicit precision configuration belongs in the
reference manifest, even when input/output arrays are FP32.
[MLX precision controls](https://ml-explore.github.io/mlx/build/html/usage/precision.html)

**Exit:** a native one-block probe with documented dtype policy, finite outputs,
and intermediate parity. Keep any mixed-precision workaround explicit.

## 4. Implement the full SS forward and verify attention memory

Use the existing token-major, channel-contiguous FP32 boundary:

| Tensor | Shape | Contract |
| --- | --- | --- |
| latent | `[1,4096,8]` | Noisy 16-cubed grid |
| timestep | `[1]` | Already scaled by 1000 |
| conditioning | `[1,1029,1024]` | Positive or negative conditioning |
| velocity | `[1,4096,8]` | Finite model prediction |

- Extend the checked block implementation to the full checkpoint. Preserve
  reference normalization, activation, modulation, RoPE, head layouts, and
  attention scaling. Cache invariant weights and positional data.
- Use MLX's optimized scaled-dot-product attention. Verify the pinned C++ API
  can require fused execution for these shapes; current documentation exposes
  `force_fused` and failure when a fused kernel is unavailable. A successful
  operator call alone is insufficient evidence of bounded scratch.
- Capture Metal execution/allocation evidence for both self- and cross-attention.
  Confirm no full `[1,12,4096,4096]` score allocation: that is 384 MiB per score
  tensor at 16-bit precision, before other buffers. If fusion is unavailable,
  investigate supported layouts or bounded attention before proceeding.
- Force evaluation and wait for the relevant work before reading outputs or
  stopping timers. MLX graph construction is lazy; timing unevaluated operations
  would understate execution time. Include boundary copies and layout conversions.

[MLX attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)

**Exit:** full-model early/middle/late fixture execution, explicit attention
memory evidence, and a measured comparison against GGML Metal on the same Mac.

## 5. Integrate MLX through the native sampler and test lifetime

- Introduce a selectable SS backend (`ggml`, `tensorrt`, `mlx`) independent of
  device selection. Preserve the existing `--ss-engine` path for compatibility;
  reject contradictory settings. Requested MLX failure must reach the caller,
  with no silent GGML fallback. Record actual per-stage precision and backend.
- Connect the runner at the existing `FlowFwd` boundary. Keep the current native
  positive/negative call order, RNG, timestep schedule, guidance, and Euler code.
- Begin with explicit host-array transfers and synchronization. Unified memory
  does not by itself establish shared buffer ownership or command dependencies
  across MLX, GGML, and Glint. Optimize copies only after measuring their cost.
- Own MLX streams and model state for the job/SS stage. Evaluate and drain work
  before releasing arrays, recycling buffers, returning cancellation, or handing
  results to another framework. Do not wait for inference on Glint's UI thread.
  Separate queues do not guarantee viewport priority or freedom from stalls.
- Release SS weights before the later model stages. Measure active allocation,
  retained allocator cache, and idle footprint separately. Define a bounded
  cache policy before introducing persistent model caching.
- Run success -> mid-step cancel -> invalid/missing weights -> success, followed
  by at least ten completed jobs. Also exercise invalid image bytes and shutdown
  during a job. Keep memory budgets fixed and check idle footprint for a trend
  after warm-up. Test cancellation at stage boundaries and after an evaluated
  forward; cancellation need not interrupt a kernel already in flight.
- Build/run CPU and CUDA contract checks with MLX disabled to protect existing
  configurations. Run native Mac tests and a Python-free packaged smoke test.

[MLX streams](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html)

**Exit:** repeated native MLX-backed textured jobs, clean failures/cancellation,
and bounded idle residency without restarting the process.

## 6. Compare full-pipeline quality, memory, and time

Use the same native decoders to isolate latent differences. Provisional numerical
gates inherited from the TensorRT experiment are relative L2 <= 2% and maximum
absolute error <= `0.02 + 0.05 * max(abs(reference))` for each captured forward.
Report pointwise errors too. Treat these as review thresholds, not proof of visual
equivalence; do not silently loosen them to make a new backend pass.

Require decoded SS occupancy IoU >= 95%, all four fixed silhouette views per
asset >= 95%, valid embedded PBR textures, and visual review of thin geometry,
disconnected parts, roughness, and metallic detail under fixed Glint lighting.
Measure GGML Metal repeat variation. The previous TensorRT knight side view was
94.59%; that exception is not automatic approval for the MLX implementation.
Report final latent error separately from per-forward error and occupancy.

For each fixture, record end-to-end time, per-stage time, load/compile time,
memory through serialization, and idle memory before/after. Compare GGML Metal
and MLX on the same Mac, with matching background work and at least three timed
repetitions after a separate cold run. Report medians and ranges.

Apple silicon's CPU and GPU share memory. Collect MLX active/cache/peak counters,
GGML buffers, Metal resource/residency information where available, process
physical footprint, system memory pressure, compression, and swap activity.
Keep overlapping counters separate; do not sum them into a fictional device-wide
VRAM total. Allocator limits do not reserve the rest of memory for the renderer.
[MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)

Carry forward **6 GiB additional memory** as an experimental target, not as a
claimed equivalent of the Windows VRAM measurement. Define the primary Mac
footprint metric, sampling interval, and baseline before testing. Report the
same metric for baseline and generation, plus the other counters separately.
Acceptance also requires no sustained generation-induced swap growth or severe
memory pressure on the named supported configuration.

**Exit:** a four-fixture report that identifies the largest remaining stage and
shows whether MLX earns its maintenance cost. Keep GGML as the default if the
end-to-end benefit is within measured variation or quality remains unresolved.

## 7. Glint Metal handoff and product acceptance

- Implement one native worker service for UI and MCP. Reuse an existing shared
  service if the CUDA handoff lands first. Pass encoded input, seed, preset,
  backend choice, progress, cancellation, and result through that service.
- Publish completed GLB bytes on the editor thread as one undoable append.
  Cancellation and failures must leave no partial scene edit. Validate undo/redo,
  material/texture bindings, job shutdown, and edits while generation is active.
- Use a dedicated test scene rather than replacing an unrelated open scene.
  Keep the viewport rendering throughout generation and asset publication.
  Measure CPU frame times and GPU timings separately: baseline and overlap
  p50/p95/p99/max, texture upload, acceleration build, and total memory pressure.
- Run GGML Metal and MLX against the same scene and render settings. Set the
  acceptable viewport latency/frame-time budget before the acceptance run.
  If contention exceeds it, test bounded submission or scheduling between
  forwards; a background thread alone does not solve GPU contention.
- Record the exact Glint build and Mac environment. Test the packaged app's MLX
  library/Metal-resource discovery without developer environment paths.

**Exit:** UI and MCP generation both work in the actual Metal editor, produce
one undoable textured asset, and meet recorded memory and responsiveness limits.
Standalone/headless results alone do not close this gate.

## Follow-up decision and deliverables

Port another stage only when the Mac measurements justify it. If shape/texture
flow dominates, capture real active-token counts and investigate its sparse
coordinate/gather/attention requirements before committing to MLX. If CPU
deformable convolution, decimation, sparse decoding, or bake dominates, target
that stage with a bounded native/Metal implementation first. Consider
quantization or tiling only against an observed memory problem.

Expected artifacts: an optional MLX C++ target/runner; explicit device and SS
backend selection; pinned conversion/fixture manifests; Mac build/package
instructions; numerical and lifetime tests; four-fixture timing/quality/memory
reports; and actual Glint Metal integration evidence. Keep weights and large
captures outside Git. Each exit check remains open until the named hardware
produces the evidence.
