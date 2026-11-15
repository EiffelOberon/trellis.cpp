"""Compare default resident BF16 with resident INT8, including full-job quality."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from benchmark_memory import run
sys.path.insert(0, str(Path(__file__).resolve().parent/'onnx'))
from compare_silhouettes import geometry, silhouette
from validate_glb import validate


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--exe', type=Path, default=Path('build-baseline/trellis-job.exe'))
    parser.add_argument('--models', type=Path, default=Path('models/f16'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = ['ss_flow_bf16.plan', 'ss_flow_int8.plan', 'shape_flow_512.gguf', 'tex_flow_512.gguf',
             'int8/shape_flow_512.gguf', 'int8/tex_flow_512.gguf']
    report = dict(schema_version=1, comparison='resident-bf16-vs-resident-int8',
                  input_sha256=sha256(args.image), executable_sha256=sha256(args.exe),
                  model_sha256={name: sha256(args.models/name) for name in paths},
                  gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip(),
                  commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  sample_ms=dict(worker_cuda=20, device_wide=50, process_rss=20),
                  note='Fresh sequential workers, seed 42, 512, 12 steps, 150K faces, 1K PNG atlas; streaming disabled in both profiles. Source may be uncommitted.')
    # No profile/engine override on the control: verify the actual new default.
    report['bf16'] = run(args, 'bf16', profile='default')
    report['int8'] = run(args, 'int8', profile='int8')
    for mode in ['bf16','int8']:
        metrics = report[mode]['worker']
        if metrics['profile'] != mode or metrics['stream_flow_blocks'] != 0 or metrics['ss_weight_budget_mib'] != -1:
            raise RuntimeError('Unexpected profile or streaming configuration')
    aroot, broot = [args.output/mode/'capture' for mode in ['bf16','int8']]
    a = np.load(aroot/'ss_logits.npy') > 0
    b = np.load(broot/'ss_logits.npy') > 0
    occupancy = float(np.count_nonzero(a & b)/max(1,np.count_nonzero(a | b)))
    finite = {}
    for name in ['ss_final','ss_logits','shape_latent','texture_latent','shape_features','texture_features']:
        finite[name] = all(bool(np.isfinite(np.load(root/f'{name}.npy')).all()) for root in [aroot,broot])
    ageom, bgeom = [geometry(args.output/mode/'asset.glb') for mode in ['bf16','int8']]
    silhouettes = []
    for angle in [0,90,180,300]:
        am, bm = silhouette(*ageom, angle), silhouette(*bgeom, angle)
        iou = float(np.count_nonzero(am & bm)/max(1,np.count_nonzero(am | bm)))
        silhouettes.append(dict(angle=angle, iou=iou, passed=iou >= .95))
        overlay = np.zeros((512,512,3), dtype=np.uint8)
        overlay[am & bm] = [210,210,210]; overlay[am & ~bm] = [240,90,90]; overlay[bm & ~am] = [70,160,250]
        Image.fromarray(overlay).save(args.output/f'silhouette-{angle}.png')
    report['quality'] = dict(ss_occupancy_iou=occupancy, passed_ss_iou_95=occupancy >= .95,
                             identical_initial_noise=bool(np.array_equal(np.load(aroot/'noise_0.npy'),np.load(broot/'noise_0.npy'))),
                             finite_captures=finite, silhouettes=silhouettes,
                             glb_validation=[validate(args.output/mode/'asset.glb') for mode in ['bf16','int8']])
    report['quality']['passed'] = (occupancy >= .95 and all(x['passed'] for x in silhouettes) and all(finite.values())
                                  and report['quality']['identical_initial_noise']
                                  and all(report[m]['worker']['repaired_nonfinite'] == 0 for m in ['bf16','int8']))
    report['quality']['scope'] = 'Geometry/finite/PBR structure checks; visual texture/material acceptance is separate. INT8 is lossy; byte parity is not expected.'
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['quality'], indent=2), flush=True)
    # Quality failures are reported without masking a successful experimental execution.
    print('QUALITY:', 'PASS' if report['quality']['passed'] else 'FAIL (experimental)', flush=True)


if __name__ == '__main__':
    main()
