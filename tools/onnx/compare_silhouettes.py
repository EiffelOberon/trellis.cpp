"""Compare fixed orthographic silhouettes of native and TensorRT mesh outputs."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw


def geometry(path):
    data = path.read_bytes()
    size = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20+size])
    binary = data[28+size:]
    primitive = gltf["meshes"][0]["primitives"][0]

    def array(index, components):
        accessor = gltf["accessors"][index]
        view = gltf["bufferViews"][accessor["bufferView"]]
        dtype = {5126: "<f4", 5125: "<u4", 5123: "<u2"}[accessor["componentType"]]
        return np.frombuffer(binary, dtype=dtype, count=accessor["count"]*components,
                             offset=view.get("byteOffset", 0)+accessor.get("byteOffset", 0)).reshape(-1,components)
    return array(primitive["attributes"]["POSITION"], 3), array(primitive["indices"], 1).reshape(-1,3)


def silhouette(positions, faces, angle):
    angle = np.radians(angle)
    x = positions[:,0]*np.cos(angle)+positions[:,2]*np.sin(angle)
    xy = np.stack(((x*.9+.5)*512, (-positions[:,1]*.9+.5)*512), axis=1).astype(np.int32)
    result = Image.new("L", (512,512))
    draw = ImageDraw.Draw(result)
    for triangle in xy[faces].reshape(-1,6).tolist():
        draw.polygon(triangle, fill=255)
    return np.array(result) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("tensorrt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-run", type=int, default=0)
    parser.add_argument("--tensorrt-run", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = []
    for fixture in ["prop", "thin", "parts", "material"]:
        a = geometry(args.baseline / f"{fixture}_{args.baseline_run}.glb")
        b = geometry(args.tensorrt / f"{fixture}_{args.tensorrt_run}.glb")
        for angle in [0,90,180,300]:
            am = silhouette(*a, angle); bm = silhouette(*b, angle)
            iou = float(np.count_nonzero(am & bm)/max(1,np.count_nonzero(am | bm)))
            metrics.append(dict(fixture=fixture, angle=angle, silhouette_iou=iou, passed_iou_95=iou>=.95))
            overlay = np.zeros((512,512,3), dtype=np.uint8)
            overlay[am & bm] = [210,210,210]
            overlay[am & ~bm] = [240,90,90]
            overlay[bm & ~am] = [70,160,250]
            Image.fromarray(overlay).save(args.output / f"{fixture}_{angle}.png")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
