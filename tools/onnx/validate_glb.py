"""Validate embedded PBR GLB structure, texture decoding, and finite mesh data."""
import argparse
import io
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image


def validate(path):
    data = path.read_bytes()
    if len(data) < 28 or struct.unpack_from("<III", data) != (0x46546C67, 2, len(data)):
        raise ValueError(f"Invalid GLB header: {path}")
    json_size, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != 0x4E4F534A:
        raise ValueError("Missing JSON chunk")
    gltf = json.loads(data[20:20+json_size])
    offset = 20 + json_size
    bin_size, chunk_type = struct.unpack_from("<II", data, offset)
    if chunk_type != 0x004E4942 or offset + 8 + bin_size != len(data):
        raise ValueError("Invalid BIN chunk")
    binary = data[offset+8:]
    if len(gltf["buffers"]) != 1 or gltf["buffers"][0]["byteLength"] > len(binary):
        raise ValueError("Invalid embedded buffer")
    for view in gltf["bufferViews"]:
        if view.get("buffer", 0) != 0 or view.get("byteOffset", 0) + view["byteLength"] > len(binary):
            raise ValueError("Buffer view exceeds embedded buffer")
    decoded = []
    for image in gltf["images"]:
        if "uri" in image:
            raise ValueError("Texture is not embedded")
        view = gltf["bufferViews"][image["bufferView"]]
        begin = view.get("byteOffset", 0)
        with Image.open(io.BytesIO(binary[begin:begin+view["byteLength"]])) as texture:
            texture.load()
            decoded.append(texture.size)
    vertices = faces = 0
    for mesh in gltf["meshes"]:
        for primitive in mesh["primitives"]:
            if primitive.get("mode", 4) != 4 or "TEXCOORD_0" not in primitive["attributes"]:
                raise ValueError("Expected textured triangles")
            position = gltf["accessors"][primitive["attributes"]["POSITION"]]
            indices = gltf["accessors"][primitive["indices"]]
            if position["componentType"] != 5126 or position["type"] != "VEC3":
                raise ValueError("Expected float32 positions")
            view = gltf["bufferViews"][position["bufferView"]]
            begin = view.get("byteOffset", 0) + position.get("byteOffset", 0)
            positions = np.ndarray((position["count"], 3), dtype="<f4", buffer=binary, offset=begin,
                                   strides=(view.get("byteStride", 12), 4))
            if not np.isfinite(positions).all() or not position["count"] or indices["count"] % 3:
                raise ValueError("Invalid mesh data")
            vertices += position["count"]; faces += indices["count"] // 3
            pbr = gltf["materials"][primitive["material"]]["pbrMetallicRoughness"]
            for name in ["baseColorTexture", "metallicRoughnessTexture"]:
                texture = gltf["textures"][pbr[name]["index"]]
                source = texture.get("source", texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source"))
                if source is None or source >= len(decoded):
                    raise ValueError("PBR texture source missing")
    return dict(path=str(path), vertices=vertices, triangles=faces, textures=decoded,
                bytes=len(data), valid=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glb", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([validate(path) for path in args.glb], indent=2))


if __name__ == "__main__":
    main()
