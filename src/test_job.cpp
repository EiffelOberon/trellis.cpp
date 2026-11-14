#include "trellis_job.h"
#include "trellis_model.h"
#include "flow_runner.h"
#include "job_control.h"
#include "mesh_glb.h"
#include "preprocess.h"
#include "ggml.h"
#include "stb_image_write.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <type_traits>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    try {
        static_assert(!std::is_copy_constructible<trellis::Model>::value, "GPU model must have one owner");
        {
            trellis::Model a;
            a.meta = ggml_init({1 << 20, nullptr, true});
            require(a.meta != nullptr, "metadata allocation failed");
            auto* original = a.meta;
            trellis::Model b(std::move(a));
            require(!a.meta && b.meta == original, "model move did not transfer ownership");
            a = std::move(b);
            require(!b.meta && a.meta == original, "model assignment did not transfer ownership");
            a.free(); a.free();
        }
        trellis::JobRequest job;
        require(trellis::run_job(job).status == trellis::JobStatus::Failed, "empty image accepted");
        std::atomic_bool cancel{true};
        require(trellis::run_job(job, &cancel).status == trellis::JobStatus::Cancelled, "pre-cancel ignored");
        cancel = false;
        trellis::RunControl control; control.cancel = &cancel;
        trellis::SamplerParams params; params.steps = 4; params.guidance_strength = 1; params.control = &control;
        int calls = 0;
        try {
            trellis::sample_flow([&](const std::vector<float>& x, float, const trellis::FlowCond&) {
                ++calls; cancel = true; return x;
            }, {1,2,3,4}, nullptr, nullptr, params);
            throw std::runtime_error("sampler did not cancel");
        } catch (const trellis::JobCancelled&) {}
        require(calls == 1, "sampler continued after cancellation");
        std::vector<unsigned char> rgba(8 * 8 * 4, 0), png;
        for (int y = 2; y < 6; ++y) for (int x = 2; x < 6; ++x) {
            rgba[(y*8+x)*4] = 100; rgba[(y*8+x)*4+3] = 255;
        }
        require(stbi_write_png_to_func([](void* context, void* bytes, int size) {
            auto& out = *static_cast<std::vector<unsigned char>*>(context);
            auto* data = static_cast<unsigned char*>(bytes); out.insert(out.end(), data, data + size);
        }, &png, 8, 8, 4, rgba.data(), 8*4) != 0, "PNG encode failed");
        trellis::ImageInput input(png.data(), png.size());
        require(trellis::image_has_alpha(input), "memory image lost alpha");
        int size = 0;
        require(!trellis::threshold_cutout(input, size).empty() && size > 0, "memory preprocessing failed");
        job.image = png; job.models = "directory-that-does-not-exist";
        require(trellis::run_job(job).status == trellis::JobStatus::Failed, "missing model directory accepted");
        const float verts[] = {0,0,0, 1,0,0, 0,1,0}, uv[] = {0,0,1,0,0,1};
        const int32_t faces[] = {0,1,2};
        std::vector<uint8_t> glb;
        require(trellis::write_glb_textured(nullptr, verts, 3, uv, faces, 1, rgba.data(), rgba.data(), 8,
                                           false, 42, nullptr, false, nullptr, nullptr, &glb), "memory GLB failed");
        require(glb.size() > 20 && !std::memcmp(glb.data(), "glTF", 4), "invalid memory GLB");
        uint32_t json_size = 0;
        std::memcpy(&json_size, glb.data()+12, sizeof(json_size));
        const std::string json(reinterpret_cast<char*>(glb.data()+20), json_size);
        require(json.find("metallicRoughnessTexture") != std::string::npos, "PBR texture missing");
        require(!trellis::write_glb("missing-directory/out.glb", verts, 3, faces, 1), "write failure reported success");
        printf("PASS: job validation, cancellation, model ownership, image bytes, PBR GLB bytes\n");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
