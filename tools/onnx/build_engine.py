"""Build a fixed-shape TensorRT engine offline and preserve inspection metadata."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import tensorrt as trt
from fetch_models import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--weight-streaming", action="store_true")
    args = parser.parse_args()
    if args.engine.exists():
        parser.error('Engine output already exists; choose a fresh path to preserve installed profile provenance')
    manifest = json.loads(args.onnx.with_suffix(".manifest.json").read_text())
    if sha256(args.onnx) != manifest["onnx_sha256"]:
        raise RuntimeError("ONNX graph hash does not match its manifest")
    for entry in manifest["external_data"]:
        if sha256(args.onnx.parent / entry["file"]) != entry["sha256"]:
            raise RuntimeError("ONNX external weights hash mismatch")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    # TensorRT 11 always uses strong typing. Precision comes from explicit graph casts.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse_from_file(str(args.onnx.resolve())):
        raise RuntimeError("\n".join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
    network.name = f"trellis2.ss_flow.v1.512.{manifest['precision']}.{'probe' if manifest['block_only'] else 'full'}"
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if args.weight_streaming:
        config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)
    start = time.monotonic()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    with args.engine.open('xb') as output:
        output.write(bytes(plan))
    del plan, config, onnx_parser, network, builder
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    inspector = engine.create_engine_inspector()
    inspection = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    args.engine.with_suffix(".layers.json").write_text(inspection)
    manifest.update(tensorrt=trt.__version__, engine_sha256=sha256(args.engine),
                    build_seconds=time.monotonic() - start,
                    workspace_limit_bytes=args.workspace_mib << 20,
                    weight_streaming=args.weight_streaming,
                    context_device_memory_bytes=engine.device_memory_size_v2,
                    gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                                                 "--format=csv,noheader"], text=True).strip(),
                    attention_memory_verified=False)
    args.engine.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
