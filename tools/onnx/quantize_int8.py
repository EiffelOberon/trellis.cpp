"""Experimental calibrated INT8 Q/DQ for SS transformer linear layers.

Preserves BF16 attention, original norm precision, and FP32 I/O. Calibration replays captured
forwards through the exported real-RoPE PyTorch model. No random calibration,
runtime dependency, or implicit TensorRT precision flag is used.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch
from safetensors.torch import load_file

from export_ss_flow import load_reference, TokenWrapper, real_rope, install_fp32_residuals
from fetch_models import sha256


def calibrate(args):
    source_manifest = json.loads(args.onnx.with_suffix('.manifest.json').read_text())
    cls = load_reference(args.reference.resolve())
    config = json.loads(Path(str(args.checkpoint) + ".json").read_text())["args"]
    model = cls(**config).eval()
    missing, unexpected = model.load_state_dict(load_file(str(args.checkpoint) + ".safetensors"), strict=False)
    if unexpected or any(name != "rope_phases" for name in missing):
        raise RuntimeError(f"Checkpoint mismatch: {missing}, {unexpected}")
    model.convert_to(torch.bfloat16)
    from trellis2.modules.attention.rope import RotaryPositionEmbedder
    model.rope_phases = torch.view_as_real(model.rope_phases).clone()
    RotaryPositionEmbedder.apply_rotary_embedding = staticmethod(real_rope)
    if source_manifest.get('fp32_residuals', False):
        install_fp32_residuals(model)
    wrapper = TokenWrapper(model).eval().cuda()
    maxima, handles = {}, []
    for name, module in wrapper.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.startswith("model.blocks."):
            continue

        def observe(layer, inputs, key=name):
            x = inputs[0].detach().to(layer.weight.dtype)
            current = x.abs().amax(dim=tuple(range(x.ndim - 1))).float()
            maxima[key] = torch.maximum(maxima[key], current) if key in maxima else current

        handles.append(module.register_forward_pre_hook(observe))
    fixtures = sorted(args.calibration.glob("*/forward_*"))
    if not fixtures:
        raise RuntimeError("Calibration root must contain ASSET/forward_XXX directories")
    provenance = []
    with torch.inference_mode():
        for index, fixture in enumerate(fixtures):
            inputs = [torch.from_numpy(np.load(fixture / f"{name}.npy")).cuda()
                      for name in ["latent", "timestep", "cond"]]
            output = wrapper(*inputs)
            if not torch.isfinite(output).all():
                raise RuntimeError(f"Nonfinite calibration output: {fixture}")
            provenance.append(dict(fixture=str(fixture), inputs={
                name: sha256(fixture / f"{name}.npy") for name in ["latent", "timestep", "cond"]}))
            print(f"calibration {index + 1}/{len(fixtures)}: {fixture}", flush=True)
    for handle in handles:
        handle.remove()
    result = dict(method="per-input-channel absolute maximum over captured forwards",
                  source_onnx_sha256=source_manifest['onnx_sha256'],
                  fp32_residuals=source_manifest.get('fp32_residuals', False),
                  checkpoint_sha256=sha256(Path(str(args.checkpoint) + ".safetensors")),
                  fixtures=provenance, channels={name: value.cpu().tolist() for name, value in maxima.items()})
    args.scales.parent.mkdir(parents=True, exist_ok=True)
    args.scales.write_text(json.dumps(result, indent=2) + "\n")
    del wrapper, model, maxima
    torch.cuda.empty_cache()
    return result


def quantize(args, calibration):
    source_manifest = json.loads(args.onnx.with_suffix(".manifest.json").read_text())
    if source_manifest["precision"] != "bf16" or sha256(args.onnx) != source_manifest["onnx_sha256"]:
        raise RuntimeError("Expected the verified BF16 source graph")
    if calibration["checkpoint_sha256"] != source_manifest["checkpoint_sha256"]:
        raise RuntimeError("Calibration checkpoint does not match source ONNX")
    if source_manifest.get('fp32_residuals', False) and (not calibration.get('fp32_residuals', False) or
            calibration.get('source_onnx_sha256') != source_manifest['onnx_sha256']):
        raise RuntimeError('FP32 residual graph requires calibration captured with the matching precision policy')
    for entry in source_manifest["external_data"]:
        if sha256(args.onnx.parent / entry["file"]) != entry["sha256"]:
            raise RuntimeError("Source external weights hash mismatch")
    model = onnx.load(args.onnx)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    added, removed, nodes, records = [], set(), [], []

    def constant(name, value):
        added.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    for node in model.graph.node:
        metadata = {entry.key: entry.value for entry in node.metadata_props}
        scopes = ast.literal_eval(metadata.get("pkg.torch.onnx.name_scopes", "[]"))
        candidates = [name for name in scopes if name in calibration["channels"]]
        if node.op_type != "MatMul" or node.input[1] not in initializers or not candidates:
            nodes.append(node)
            continue
        module = candidates[-1]
        if args.include and not any(pattern in module for pattern in args.include):
            nodes.append(node)
            continue
        if getattr(args, "exclude", None) and any(pattern in module for pattern in args.exclude):
            nodes.append(node)
            continue
        weight_tensor = initializers[node.input[1]]
        if weight_tensor.data_type != onnx.TensorProto.BFLOAT16:
            raise RuntimeError(f"Expected BF16 transformer weight: {module}")
        # Exported MatMul weight is [input, output], unlike PyTorch Linear.
        bf16 = numpy_helper.to_array(weight_tensor).dtype
        weight = numpy_helper.to_array(weight_tensor).astype(np.float32)
        activation = np.asarray(calibration["channels"][module], dtype=np.float32)
        if weight.ndim != 2 or weight.shape[0] != len(activation) or not np.isfinite(activation).all():
            raise RuntimeError(f"Bad calibration dimensions/values: {module}")
        prefix = node.name + "_int8"
        x = node.input[0]
        if args.smooth_alpha is not None:
            smooth = (np.maximum(activation, 1e-5) ** args.smooth_alpha /
                      np.maximum(np.abs(weight).max(axis=1), 1e-5) ** (1 - args.smooth_alpha))
            smooth = np.clip(smooth, 1e-3, 1e3).astype(bf16).astype(np.float32)
            s = constant(prefix + "_smooth", smooth.astype(bf16))
            nodes.append(helper.make_node("Div", [x, s], [prefix + "_smoothed"], name=prefix + "_smooth_div"))
            x = prefix + "_smoothed"
            weight *= smooth[:, None]
            activation /= smooth
        # Round scales to the DQ computation type before producing INT8 weights.
        w_scale = np.maximum(np.abs(weight).max(axis=0) / 127, 1e-10).astype(bf16)
        x_scale = np.asarray(max(float(activation.max()) / 127, 1e-10), dtype=bf16)
        quantized = np.clip(np.rint(weight / w_scale.astype(np.float32)), -127, 127).astype(np.int8)
        w = constant(prefix + "_weight", quantized)
        ws = constant(prefix + "_weight_scale", w_scale)
        wz = constant(prefix + "_weight_zero", np.zeros(weight.shape[1], dtype=np.int8))
        xs = constant(prefix + "_input_scale", x_scale)
        xz = constant(prefix + "_input_zero", np.asarray(0, dtype=np.int8))
        nodes += [helper.make_node("QuantizeLinear", [x, xs, xz], [prefix + "_xq"], name=prefix + "_x_quant"),
                  helper.make_node("DequantizeLinear", [prefix + "_xq", xs, xz], [prefix + "_xdq"], name=prefix + "_x_dequant"),
                  helper.make_node("DequantizeLinear", [w, ws, wz], [prefix + "_wdq"], axis=1, name=prefix + "_w_dequant")]
        removed.add(node.input[1])
        node.input[0], node.input[1] = prefix + "_xdq", prefix + "_wdq"
        nodes.append(node)
        records.append(dict(module=module, node=node.name, weights=int(weight.size), input_scale=float(x_scale)))
    if not records:
        raise RuntimeError("No transformer linear layers matched")
    # Drop replaced constants only when no remaining node references them.
    used = {name for node in nodes for name in node.input}
    kept = [tensor for name, tensor in initializers.items() if name not in removed or name in used]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept + added)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    external = args.output.name + ".data"
    if args.output.exists() or (args.output.parent / external).exists():
        raise RuntimeError("Output already exists; choose a new output path")
    onnx.save_model(model, args.output, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=external, size_threshold=1024)
    onnx.checker.check_model(str(args.output))
    manifest = dict(source_manifest)
    manifest.update(precision="int8-bf16", source_onnx_sha256=source_manifest["onnx_sha256"],
                    onnx_sha256=sha256(args.output), calibration_sha256=sha256(args.scales),
                    quantization=dict(activation="symmetric per-tensor minmax INT8", weight="symmetric per-output-channel INT8",
                                      high_precision="BF16", smooth_alpha=args.smooth_alpha, layers=records,
                                      fixtures=calibration["fixtures"], experimental=True),
                    external_data=([dict(file=external, sha256=sha256(args.output.parent / external))]
                                   if (args.output.parent / external).exists() else []))
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Quantized {len(records)} linears, {sum(x['weights'] for x in records):,} weights -> {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True, help="Cache of calibrated per-channel maxima")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth-alpha", type=float, help="Optional SmoothQuant-style input/weight balancing in [0,1]")
    parser.add_argument("--include", action="append", help="Only quantize modules containing one of these substrings")
    parser.add_argument("--exclude", action="append", help="Keep modules containing one of these substrings in BF16")
    args = parser.parse_args()
    if args.smooth_alpha is not None and not 0 <= args.smooth_alpha <= 1:
        parser.error("smooth-alpha must be in [0,1]")
    calibration = json.loads(args.scales.read_text()) if args.scales.exists() else calibrate(args)
    quantize(args, calibration)


if __name__ == "__main__":
    main()
