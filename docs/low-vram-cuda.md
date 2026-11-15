# CUDA memory optimization without quantization

Historical weight-streaming experiment. Current configuration uses
[resident BF16 by default and an experimental resident INT8 optimized profile](resident-profiles.md).
Neither current named profile uses weight streaming. The figures below apply
only to this separate streaming experiment.

This adds an opt-in, lower-VRAM execution mode for the native 512 worker on
the RTX 5080. SS stays BF16; sparse shape and texture weights stay FP16,
with the existing BF16 attention KV and FP32 attention accumulation.
INT8 is not required. The resident execution mode remains available.

## Changes

- Sparse DiT weight streaming: keep the full checkpoint on the CPU, pack
  groups into pinned host buffers, and reuse one GPU weight buffer and one
  transformer graph. Hidden states remain on the GPU. Group sizes must divide
  30; `5` is the tested lower-memory setting.
- TensorRT SS weight streaming: an engine built with `--weight-streaming`
  accepts `--ss-weight-budget-mib` before context creation. The tested budget
  is 1024 MiB. A budget on an ordinary engine fails explicitly.
- CUDA decoder working sets: bound the widened ConvNeXt intermediate to
  128 MiB and local linear graphs to approximately 64 MiB. Write disjoint
  ConvNeXt output ranges into one allocation instead of a growing concat.
  These are workspace targets, not limits on total device memory. Existing
  chunk-count safeguards can require larger chunks for large inputs.
- Pad a short ConvNeXt tail to the full chunk width before GEMMs, then crop
  it. Short tails otherwise select different CUDA half-precision GEMM
  algorithms and can accumulate material decoder drift. The regression
  exercises one full tile plus an odd-length tail, including four blocks.
  Local linear projections use the same padding policy: a chest fixture
  exposed a smaller rounding change confined to the final 24,365 shape-head
  rows before this was added.

The smaller decoder defaults apply only to CUDA. Other backends retain their
previous budgets and concatenation path. Sparse DiT streaming currently
requires direct CUDA execution, without the multi-backend scheduler. Large
cascade stages retain their existing streamed-decoder fallback; 1024 and
1536 presets have not been accepted by this experiment.

## Build and use

```powershell
.\scripts\build-onnx-windows.ps1 -BuildDir build-baseline -Test
$env:PATH="$PWD\.venv\Lib\site-packages\tensorrt_libs;$env:CUDA_PATH\bin\x64;$env:PATH"

.\.venv\Scripts\python.exe tools/onnx/build_engine.py `
  out/onnx-tensorrt/ss_flow.onnx out/optimization/ss_flow_stream.plan `
  --weight-streaming

.\build-baseline\trellis-job.exe IMAGE.png OUTPUT.glb models/f16 `
  --profile native `
  --ss-engine out/optimization/ss_flow_stream.plan `
  --ss-weight-budget-mib 1024 --stream-flow-blocks 5
```

The same options are available on `trellis-cli`. Library callers set
`JobRequest::ss_weight_budget_mib = 1024` and `stream_flow_blocks = 5`.
Defaults are `-1` (engine default budget) and `0` (resident sparse flows).
Model loading, transfers, decoder readback, mesh processing, baking and GLB
serialization are all included in worker timings.

## Verification

```powershell
.\build-baseline\trellis-test-streamed-flow.exe models/f16
.\build-baseline\trellis-test-sparse-chunk.exe models/f16/shape_dec.gguf

.\.venv\Scripts\python.exe tools/benchmark_memory.py IMAGE.png `
  --stream-engine out/optimization/ss_flow_stream.plan --output out/memory-check
```

The comparison launches fresh processes sequentially. Both use the same
streaming-capable BF16 engine; the control leaves all weights resident and
uses the old decoder workspace budgets. This isolates execution and residency
from engine rebuild differences. SS latents/logits, shape/texture latents,
and decoded shape/PBR features are checked before mesh processing. The gate
requires matching shapes, finite data and relative L2 <= 1e-5.

Rebuilding a TensorRT engine can change BF16 tactics independently of streaming.
An earlier comparison against the original engine had different SS latents
on the sofa, but identical downstream shape/texture latents and decoded
features. It is not used as evidence of byte-exact SS execution.

The real-checkpoint flow regression checks shape and texture models at four
forwards with changing timesteps and positive/zero conditioning, for groups
of 1, 5 and 30 blocks. All 24 outputs were byte-identical to resident execution.
The decoder regression checks full/tail chunks across one and four blocks.
CPU-only and CUDA worker contract tests also pass.

The rebuilt streaming-capable SS engine also passes the four original
PyTorch reference-forward gates (relative L2 0.515%, 0.615%, 0.606%, 1.068%;
the gate is 2% plus the existing maximum-error bound). This is distinct from
the byte-identical same-engine resident/streamed comparison. Pointwise
allclose is not true for all reference forwards.

## Measurement scope and tradeoff

Worker memory is sampled `cudaMemGetInfo` usage at 20 ms, including the
machine's CUDA-visible idle baseline. Additional usage subtracts that baseline.
The benchmark separately records device-wide `nvidia-smi` usage at 50 ms
and process RSS at 20 ms; these counters are not interchangeable on WDDM.
Neither includes a subsequent Blender render or in-process Glint upload/build.

Streaming trades system RAM and PCIe transfers for lower VRAM. The full CPU
checkpoint and pinned group packs coexist during a sparse flow, so host memory
increases. Results are single paired runs, not statistical latency estimates.
Graph/decoder parity does not imply byte-identical GLBs: GPU mesh decimation
and downstream processing can vary independently.

The streaming-capable TensorRT engine can retain a CPU weight copy even with
its GPU budget left fully resident. Consequently, the paired control's RSS
already includes that CPU copy. Similar RSS between the two paired modes
does not mean that switching from the original non-streaming engine has no
host-memory cost.

## RTX 5080 results

Native Windows, driver 617.14, TensorRT 11.4.0.106, seed 42. Both variants use
the same BF16 engine, original FP16 GGUF models, 12 steps, 150K target faces,
and a 1024-pixel PNG PBR atlas. CUDA-visible idle usage was 1.260 GiB.

| Asset / measurement | Resident control | Optimized streaming |
|---|---:|---:|
| Sofa: CUDA-accounted full-job peak | 4.336 GiB | 3.098 GiB |
| Sofa: additional above CUDA idle | 3.076 GiB | 1.838 GiB |
| Sofa: device-wide peak (`nvidia-smi`) | 5.997 GiB | 4.759 GiB |
| Sofa: worker time | 38.02 s | 40.90 s |
| Sofa: peak process RSS | 5.493 GiB | 5.554 GiB |
| Chest: CUDA-accounted full-job peak | 4.590 GiB | 3.243 GiB |
| Chest: additional above CUDA idle | 3.330 GiB | 1.982 GiB |
| Chest: device-wide peak (`nvidia-smi`) | 6.250 GiB | 4.903 GiB |
| Chest: worker time | 61.79 s | 64.72 s |
| Chest: peak process RSS | 5.565 GiB | 5.627 GiB |

Additional CUDA-accounted peak usage falls by 40.3% on the sofa and 40.5% on
the chest. The cost is about 2.9 seconds per job in these single paired runs.
All six captured arrays are byte-identical for each asset: SS latent/logits,
shape/texture latents, and decoded shape/PBR features. All four emitted GLBs
pass structural, finite-geometry and embedded-PBR-texture checks. Both runs
have zero repaired non-finite values and four guidance clamps, matching their
controls.

Machine-readable reports: [sofa](benchmarks/low-vram-cuda-5080/sofa.json),
[chest](benchmarks/low-vram-cuda-5080/chest.json),
[engine provenance](benchmarks/low-vram-cuda-5080/engine-manifest.json), and
[validation](benchmarks/low-vram-cuda-5080/validation.json). The reports record
the tested executable and engine hashes; the source base is `45edf32` with
uncommitted changes. Engine files, checkpoints, full captures and generated
assets remain outside Git under `out/` and `models/`.

The optimized path is opt-in. The next measured costs are mesh processing
(about 11 s sofa / 16 s chest) and decoder residency. Glint viewport contention,
upload/build peaks, repeated long-lived worker jobs and other GPUs/presets
remain separate acceptance work.
