# Resident BF16 default and experimental INT8 optimization

The Windows CUDA 512 worker defaults to **BF16 SS with GPU-resident weights**.
`--profile int8` selects the experimental optimized path. Both named profiles
reject weight streaming, including an SS engine built with the streaming flag.

The profile names describe the flow precision policy, not every operation in
the pipeline:

| Component | Default `bf16` | Optimized `int8` |
|---|---|---|
| Sparse-structure flow | TensorRT BF16 | Existing calibrated mixed INT8/BF16 TensorRT engine |
| Shape and texture flow matmul weights | Original FP16 GGUF | Selective INT8 Q8_0 GGUF |
| Sparse flow attention KV / accumulation | BF16 / FP32 | BF16 / FP32 |
| Normalization, biases and small latent projections | Original floating point | Original floating point |
| Image conditioning, background removal and decoders | Original precision | Original precision |
| Weight residency | GPU throughout each model stage | GPU throughout each model stage |

The INT8 SS engine quantizes the first 20 transformer blocks and retains the
last 10 at BF16, as documented in the [SS quantization experiment](int8-tensorrt.md).
The two native sparse flows each quantize 214 linear weight matrices and
preserve 426 other tensors. Their files shrink from approximately 2.59 GB to
1.38 GB each. Q8_0 is weight quantization; its native GGML CUDA arithmetic
policy differs from TensorRT's calibrated W8A8 Q/DQ execution.

This fixes the limitation of the SS-only INT8 experiment: shape and texture
weights are also smaller. The decoder working-set improvements remain enabled
for both profiles. There are no reduced sampling steps or reduced output settings.

## Prepare and run

Build the native CUDA/TensorRT worker as described in
[the native validation report](onnx-tensorrt.md), and build the BF16 and mixed
INT8 engines offline for the target GPU. Neither engine should use
`--weight-streaming`. Then install the profiles:

```powershell
$env:PATH="$PWD\.venv\Lib\site-packages\tensorrt_libs;$env:CUDA_PATH\bin\x64;$env:PATH"

.\.venv\Scripts\python.exe tools/prepare_resident_profiles.py `
  --models models/f16 `
  --bf16-engine out/onnx-tensorrt/ss_flow.plan `
  --int8-engine out/int8/ss_flow_int8_mixed.plan

# Default: resident BF16 SS, original native flow weights.
.\build-baseline\trellis-job.exe IMAGE.png OUTPUT.glb models/f16

# Experimental optimized: resident INT8 across all three flows.
.\build-baseline\trellis-job.exe IMAGE.png OUTPUT-int8.glb models/f16 --profile int8
```

Preparation installs engine files as `ss_flow_bf16.plan` and
`ss_flow_int8.plan` in the selected model directory, using hard links where
possible. It creates `int8/shape_flow_512.gguf` and
`int8/tex_flow_512.gguf`; original model files remain untouched. The
`resident-profiles.json` manifest records hashes and tensor counts. Differing
existing engines are not silently overwritten.

Library callers use `JobRequest::profile`, which defaults to `"bf16"`;
set it to `"int8"` to request the optimized path. Streaming options are rejected
for both profiles. The worker JSON records the requested profile, sparse flow
streaming count (zero), and SS streaming budget (-1, with a non-streaming engine).

The general CLI also accepts `--profile bf16|int8` with `--res 512`.
On CUDA/TensorRT builds, a 512 TRELLIS.2 invocation without an explicit engine
defaults to the BF16 profile. Other presets/backends retain their native path.
`--profile native` explicitly restores manual legacy backend/model selection;
it is not the default worker configuration.

## Quantization and validation

The GGUF quantizer now requires a `.weight` name as well as a suitable
two-dimensional shape. A shape-only rule also quantized head-wise RMSNorm
gamma tensors, which are consumed by elementwise multiplication. The regression
test preserves those parameters, modulation tensors and small-K projections.

```powershell
.\.venv\Scripts\python.exe tools/test_quantize_gguf.py
.\.venv\Scripts\python.exe tools/benchmark_resident_profiles.py IMAGE.png `
  --models models/f16 --output out/resident-int8/comparison
```

The benchmark runs the actual default worker with no profile or engine override,
then `--profile int8`, sequentially. It records both CUDA-accounted memory and
device-wide `nvidia-smi` memory separately, plus process RSS, timings, engine/model
hashes, finite captures, SS occupancy IoU, four fixed silhouettes, and PBR GLB
validation. Both runs use seed 42, 512 resolution, 12 steps, 150K target faces
and a 1K PNG atlas. INT8 is lossy, so byte parity is not the acceptance test.

The geometry checks retain the earlier 95% SS occupancy and per-view silhouette
IoU thresholds. Texture/material appearance needs separate visual review.
The previous SS-only engine failed some full-trajectory gates; its isolated
forward checks alone do not establish acceptance of this larger INT8 profile.
Until complete-job quality passes, BF16 remains the default and INT8 remains
experimental.

## RTX 5080 measurements

Single paired runs on driver 617.14 with TensorRT 11.4.0.106. Both modes use
the current bounded decoder buffers and fully resident weights. CUDA-visible
idle usage was 1.260 GiB; these generation measurements precede Blender.

| Asset / metric | Default BF16 profile | Resident INT8 profile |
|---|---:|---:|
| Sofa: CUDA-accounted full-job peak | 4.200 GiB | 3.278 GiB |
| Sofa: additional above CUDA idle | 2.939 GiB | 2.018 GiB |
| Sofa: device-wide VRAM peak | 5.917 GiB | 4.995 GiB |
| Sofa: worker time | 38.35 s | 35.40 s |
| Sofa: peak process RSS | 3.143 GiB | 2.417 GiB |
| Chest: CUDA-accounted full-job peak | 4.258 GiB | 3.332 GiB |
| Chest: additional above CUDA idle | 2.998 GiB | 2.072 GiB |
| Chest: device-wide VRAM peak | 5.976 GiB | 5.050 GiB |
| Chest: worker time | 62.12 s | 60.86 s |
| Chest: peak process RSS | 3.214 GiB | 3.034 GiB |

The sofa saves 0.922 GiB at the full-job CUDA peak (22.0%, or 31.4% of the
additional allocation above idle), and is about 3 seconds faster. The INT8
SS stage now sets its peak; the native sparse flows no longer hide the SS
weight saving. Chest saves 0.926 GiB at the full-job CUDA peak. These are
single-run timings, not statistical speedup claims.

| Geometry check | Sofa | Chest | Required |
|---|---:|---:|---:|
| SS occupancy IoU | 93.26% | 91.15% | 95% |
| Minimum of four final-mesh silhouette IoUs | 96.99% | 97.98% | 95% |

Both jobs complete with zero non-finite repairs and four guidance clamps,
matching their BF16 controls. All four PBR GLBs pass structural and finite-data
checks. Every silhouette passes, but the existing mixed SS engine still misses
the occupancy gate on both images. **Overall quality acceptance is therefore
false**, and INT8 remains experimental. The threshold was not relaxed.

Both sofa outputs were rendered in Blender 5.1.1 Eevee, 1920x1280 at 256
samples, using the same saved studio camera, lights, material parameters and
render-only cleanup. Their local outputs are under
`out/resident-int8/renders/{bf16,int8}`. Cushion shape and foot color differ;
the renders are not asserted to be identical.

Reports: [sofa](benchmarks/resident-int8-5080/sofa.json),
[chest](benchmarks/resident-int8-5080/chest.json),
[model provenance](benchmarks/resident-int8-5080/models.json), and
[validation](benchmarks/resident-int8-5080/validation.json).
The reports identify the tested executable and model hashes; generated models,
captures, GLBs and scenes remain outside Git. CPU/CUDA contract tests and the
quantizer regression pass. A default BF16 request supplied with a streaming
engine is explicitly rejected before inference.

Historical weight-streaming results in
[the CUDA memory experiment](low-vram-cuda.md) are separate and do not describe
either current named profile. Repeated long-lived workers, other seeds/devices,
and integrated Glint upload/build/viewport memory remain unverified here.
