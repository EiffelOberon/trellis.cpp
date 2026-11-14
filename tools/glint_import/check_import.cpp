// Exercise Glint's actual in-memory GLB importer without changing a live editor document.
#include <glint_io.h>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: trellis-check-glint ASSET.glb [...]\n"); return 2; }
    try {
        for (int i=1; i<argc; ++i) {
            std::ifstream file(argv[i], std::ios::binary | std::ios::ate);
            if (!file || file.tellg() <= 0) throw std::runtime_error("Cannot read GLB");
            std::vector<std::byte> bytes(static_cast<size_t>(file.tellg()));
            file.seekg(0);
            if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) throw std::runtime_error("Short GLB read");
            glint::io::SourceContext context; context.virtualPath = argv[i];
            auto document = glint::io::loadScene(std::span<const std::byte>(bytes), glint::io::SceneFormat::Glb, context);
            if (document.meshes.empty() || document.materials.empty() || document.images.size()<2)
                throw std::runtime_error("Glint imported an incomplete textured asset");
            bool pbr = false;
            for (const auto& material : document.materials) {
                pbr |= material.baseColorTexture.valid() && (material.metallicRoughnessTexture.valid() ||
                    (material.metallicTexture.valid() && material.roughnessTexture.valid()));
            }
            if (!pbr) throw std::runtime_error("Glint did not preserve the base-color / metallic / roughness texture bindings");
            printf("PASS: %s meshes=%zu materials=%zu embedded_images=%zu PBR bindings preserved\n",
                   argv[i], document.meshes.size(), document.materials.size(), document.images.size());
        }
        return 0;
    } catch (const std::exception& error) { fprintf(stderr, "%s\n", error.what()); return 1; }
}
