"""Install the resident BF16/INT8 engines and create selective Q8_0 sparse flows.

Engine files must already have been built offline for the target GPU. Existing
matching engines are reused; differing engines are never silently overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import gguf

from quantize_gguf import quantize_file


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def install(source, target):
    if target.exists():
        if not os.path.samefile(source, target) and sha256(source) != sha256(target):
            raise RuntimeError(f'Existing engine differs: {target}')
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def engine_provenance(path, precision):
    manifest_path = path.with_suffix('.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('engine_sha256') != sha256(path) or manifest.get('precision') != precision or
            manifest.get('preset') != 512 or manifest.get('weight_streaming') or
            not manifest.get('attention_memory_verified')):
        raise RuntimeError(f'Engine provenance/precision/attention gate has not passed: {path}')
    result = {key:manifest[key] for key in (
        'engine_sha256','onnx_sha256','checkpoint_sha256','precision','tensorrt','gpu',
        'fp32_residuals','context_device_memory_bytes','attention_evidence') if key in manifest}
    if 'quantization' in manifest:
        quantization = manifest['quantization']
        layers = quantization['layers']
        result['quantization'] = dict(layers=len(layers), weights=sum(layer['weights'] for layer in layers),
                                      modules=[layer['module'] for layer in layers], smooth_alpha=quantization.get('smooth_alpha'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=Path('models/f16'))
    parser.add_argument('--bf16-engine', type=Path, default=Path('out/onnx-tensorrt/ss_flow.plan'))
    parser.add_argument('--int8-engine', type=Path, default=Path('out/int8/ss_flow_int8_mixed.plan'))
    args = parser.parse_args()
    provenance = dict(bf16=engine_provenance(args.bf16_engine, 'bf16'),
                      int8=engine_provenance(args.int8_engine, 'int8-bf16'))
    install(args.bf16_engine, args.models/'ss_flow_bf16.plan')
    install(args.int8_engine, args.models/'ss_flow_int8.plan')
    target_dir = args.models/'int8'
    target_dir.mkdir(exist_ok=True)
    flows = {}
    for name in ['shape_flow_512.gguf','tex_flow_512.gguf']:
        source, target = args.models/name, target_dir/name
        if not target.exists():
            quantize_file(source, target, 'Q8_0')
        reader = gguf.GGUFReader(target)
        tensors = reader.tensors
        q8 = [t for t in tensors if t.tensor_type == gguf.GGMLQuantizationType.Q8_0]
        if not q8 or any(not t.name.endswith('.weight') for t in q8):
            raise RuntimeError(f'Invalid selective INT8 weights; regenerate {target}')
        flows[name] = dict(source_sha256=sha256(source), quantized_sha256=sha256(target),
                           source_bytes=source.stat().st_size, quantized_bytes=target.stat().st_size,
                           quantized_matrices=len(q8), preserved_tensors=len(tensors)-len(q8))
    manifest = dict(schema_version=1, default_profile='bf16', optimized_profile='int8', weight_streaming=False,
                    bf16_engine_sha256=sha256(args.models/'ss_flow_bf16.plan'),
                    int8_engine_sha256=sha256(args.models/'ss_flow_int8.plan'), flows=flows, engine_provenance=provenance,
                    precision_note='BF16 SS default with original FP16 native stages; mixed INT8/BF16 SS and Q8_0 sparse flow matmuls in optimized mode. Norms, decoders and small projections retain floating point.')
    (args.models/'resident-profiles.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
