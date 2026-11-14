"""Wrap a generated GLB in a fixed camera and three emissive area lights for Glint QA."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np


def make_fixture(source, output, angle):
    data = source.read_bytes()
    size = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20+size])
    binary = bytearray(data[28+size:])
    root_nodes = gltf["scenes"][gltf.get("scene", 0)]["nodes"]
    radians = np.radians(angle)
    rig = len(gltf["nodes"])
    gltf["nodes"].append(dict(name="Asset QA rotation", children=root_nodes,
                              rotation=[0, float(np.sin(radians/2)), 0, float(np.cos(radians/2))]))
    roots = [rig]

    def accessor(array, kind, component):
        while len(binary) % 4:
            binary.append(0)
        view = len(gltf["bufferViews"])
        gltf["bufferViews"].append(dict(buffer=0, byteOffset=len(binary), byteLength=array.nbytes))
        binary.extend(array.tobytes())
        result = len(gltf["accessors"])
        entry = dict(bufferView=view, componentType=component, count=len(array), type=kind)
        if kind == "VEC3":
            entry.update(min=array.min(0).tolist(), max=array.max(0).tolist())
        gltf["accessors"].append(entry)
        return result

    gltf.setdefault("extensionsUsed", []).append("KHR_materials_emissive_strength")
    for name, center, strength in [("QA key", [1.2,1.6,1.5], 8),
                                   ("QA fill", [-1.5,.8,1.0], 4),
                                   ("QA rim", [.2,1.,-1.5], 6)]:
        center = np.array(center, dtype=np.float32)
        normal = -center / np.linalg.norm(center)
        tangent = np.cross(normal, [0,1,0]); tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        vertices = np.array([center + x*tangent*.6 + y*bitangent*.6
                             for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]], dtype="<f4")
        positions = accessor(vertices, "VEC3", 5126)
        normals = accessor(np.tile(normal, (4,1)).astype("<f4"), "VEC3", 5126)
        indices = accessor(np.array([0,1,2,0,2,3], dtype="<u2"), "SCALAR", 5123)
        material = len(gltf["materials"])
        gltf["materials"].append(dict(name=name, doubleSided=True, emissiveFactor=[1,1,1],
            extensions={"KHR_materials_emissive_strength": {"emissiveStrength": strength}},
            pbrMetallicRoughness=dict(baseColorFactor=[1,1,1,1], metallicFactor=0, roughnessFactor=1)))
        mesh = len(gltf["meshes"])
        gltf["meshes"].append(dict(name=name, primitives=[dict(attributes={"POSITION": positions, "NORMAL": normals},
                                                             indices=indices, material=material)]))
        roots.append(len(gltf["nodes"]))
        gltf["nodes"].append(dict(name=name, mesh=mesh))
    position = np.array([1.1,.45,1.8])
    back = position / np.linalg.norm(position)
    right = np.cross([0,1,0], back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    matrix = np.eye(4); matrix[:3,:3] = np.stack([right,up,back], axis=1); matrix[:3,3] = position
    gltf["cameras"] = [dict(name="Fixed QA camera", type="perspective", perspective=dict(yfov=.65, aspectRatio=16/9, znear=.01, zfar=100))]
    roots.append(len(gltf["nodes"]))
    gltf["nodes"].append(dict(name="Fixed QA camera", camera=0, matrix=matrix.flatten(order="F").tolist()))
    gltf["scenes"] = [dict(nodes=roots)]; gltf["scene"] = 0
    gltf["buffers"][0]["byteLength"] = len(binary)
    while len(binary) % 4:
        binary.append(0)
    encoded = json.dumps(gltf, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    result = struct.pack("<III", 0x46546C67, 2, 28+len(encoded)+len(binary))
    result += struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
    result += struct.pack("<II", len(binary), 0x004E4942) + binary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--angle", type=float, default=0)
    args = parser.parse_args()
    make_fixture(args.source, args.output, args.angle)


if __name__ == "__main__":
    main()
