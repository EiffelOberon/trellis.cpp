// Image preprocessing for TRELLIS.2: load -> white-bg removal -> square crop ->
// premultiply on black -> resize -> ImageNet normalize -> [3,S,S] torch CHW.
#pragma once
#include <vector>
#include <string>
#include <cstddef>

namespace trellis {
struct Model;
struct ImageInput {
    std::string path;
    const unsigned char* bytes = nullptr;
    size_t size = 0;
    ImageInput(const std::string& file) : path(file) {}
    ImageInput(const char* file) : path(file) {}
    ImageInput(const unsigned char* data, size_t length) : bytes(data), size(length) {}
    const char* c_str() const { return bytes ? "<memory>" : path.c_str(); }
};
// Returns [3*S*S] float in torch CHW order (== ggml [S,S,3,1]), or empty on failure.
std::vector<float> preprocess_image(const ImageInput& path, int S = 512);

// Non-neural background removal: existing alpha if present, else white-bg threshold;
// bbox-crop + premultiply on transparent black -> square RGBA uint8 cutout (size in `sz`).
std::vector<unsigned char> threshold_cutout(const ImageInput& path, int& sz);

// True when the image carries a real (not all-opaque) alpha channel, i.e. it is
// pre-matted and needs no background removal.
bool image_has_alpha(const ImageInput& path);

// BiRefNet background removal: run the matte, restore it to the source aspect ratio, then
// bbox-crop and premultiply -> square RGBA uint8 cutout
// (size returned in `sz`). bm = loaded birefnet model, gpu = device for the deform kernel.
std::vector<unsigned char> birefnet_cutout(const ImageInput& path, const Model& bm, int gpu, int& sz);
// Resize a square RGB/RGBA-uint8 cutout to SxS, ImageNet-normalize -> [3,S,S] torch CHW.
std::vector<float> normalize_cutout(const std::vector<unsigned char>& rgb, int sz, int S);

// Resize a square RGB-uint8 cutout to SxS and scale to [0,1] WITHOUT ImageNet normalization
// -> [3,S,S] torch CHW. The NAF guide branch (Pixal3D) consumes the raw [0,1] image, not the
// normalized one that feeds DINOv3.
std::vector<float> cutout_to_chw01(const std::vector<unsigned char>& rgb, int sz, int S);
} // namespace trellis
