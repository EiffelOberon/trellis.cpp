#include "trt_flow.h"
#include <stdexcept>
#ifdef TRELLIS_USE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace trellis {
namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("TensorRT CUDA: ") + cudaGetErrorString(status));
}
struct Logger : nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) fprintf(stderr, "[TensorRT] %s\n", message);
    }
};
struct DeviceBuffer {
    void* data = nullptr;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void allocate(size_t bytes) { check_cuda(cudaMalloc(&data, bytes)); }
};
// Load the runtime rather than link an import library: NVIDIA's Python runtime
// distribution supplies DLLs but no .lib. Headers must match the installed SDK.
struct RuntimeLibrary {
#ifdef _WIN32
    HMODULE handle = nullptr;
    ~RuntimeLibrary() { if (handle) FreeLibrary(handle); }
    void* symbol(const char* name) { return reinterpret_cast<void*>(GetProcAddress(handle, name)); }
#else
    void* handle = nullptr;
    ~RuntimeLibrary() { if (handle) dlclose(handle); }
    void* symbol(const char* name) { return dlsym(handle, name); }
#endif
    RuntimeLibrary() {
#ifdef _WIN32
        const std::wstring name = L"nvinfer_" + std::to_wstring(NV_TENSORRT_MAJOR) + L".dll";
        HMODULE owner = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&check_cuda), &owner))
            throw std::runtime_error("Cannot locate the Trellis runtime module");
        std::wstring location(32768, L'\0');
        const DWORD size = GetModuleFileNameW(owner, location.data(), static_cast<DWORD>(location.size()));
        if (!size || size >= location.size()) throw std::runtime_error("Cannot resolve the Trellis runtime directory");
        location.resize(size);
        const auto module_path = std::filesystem::path(location);
        const auto beside_module = module_path.parent_path() / name;
        if (std::filesystem::is_regular_file(beside_module))
            handle = LoadLibraryExW(beside_module.c_str(), nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        // Preserve the standalone CLI's documented PATH setup. The in-process
        // SDK always loads its packaged runtime beside the worker DLL.
        else if (module_path.extension() != L".dll") handle = LoadLibraryW(name.c_str());
        if (!handle) throw std::runtime_error("Cannot load matching TensorRT runtime beside the worker DLL (standalone CLI also accepts PATH)");
#else
        const std::string name = "libnvinfer.so." + std::to_string(NV_TENSORRT_MAJOR);
        handle = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!handle) throw std::runtime_error("Cannot load TensorRT runtime; add its library directory to PATH/LD_LIBRARY_PATH");
    }
};
}

struct TrtFlowRunner::Impl {
    RuntimeLibrary library; // outlives every object created by the runtime
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::array<DeviceBuffer, 4> buffers;
    cudaStream_t stream = nullptr;
    int device;
    std::string precision;
    ~Impl() {
        cudaSetDevice(device);
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
        // context and engine are destroyed before runtime and library by reverse
        // member order. Device buffers have no outstanding users after sync.
    }
    explicit Impl(int index) : device(index) {}
};

TrtFlowRunner::TrtFlowRunner(const std::string& path, int device, bool allow_probe, int weight_budget_mib, bool require_resident_engine) : impl_(std::make_unique<Impl>(device)) {
    if (device < 0) throw std::invalid_argument("TensorRT requires a CUDA device");
    if (weight_budget_mib < -1) throw std::invalid_argument("Invalid TensorRT weight budget");
    auto& p = *impl_;
    check_cuda(cudaSetDevice(device));
    using CreateRuntime = void* (*)(void*, int32_t);
    auto create = reinterpret_cast<CreateRuntime>(p.library.symbol("createInferRuntime_INTERNAL"));
    if (!create) throw std::runtime_error("TensorRT runtime entry point missing");
    p.runtime.reset(static_cast<nvinfer1::IRuntime*>(create(&p.logger, NV_TENSORRT_VERSION)));
    if (!p.runtime) throw std::runtime_error("TensorRT runtime version mismatch");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open TensorRT engine: " + path);
    const auto size = file.tellg();
    if (size <= 0) throw std::runtime_error("Empty TensorRT engine: " + path);
    std::vector<char> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(bytes.data(), size)) throw std::runtime_error("Cannot read TensorRT engine: " + path);
    p.engine.reset(p.runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!p.engine) throw std::runtime_error("TensorRT engine deserialization failed: " + path);
    if (require_resident_engine && p.engine->getStreamableWeightsSize() > 0)
        throw std::runtime_error("Resident profile requires an engine built without weight streaming");
    const std::string name = p.engine->getName();
    const std::string prefix = "trellis2.ss_flow.v1.512.";
    bool recognized = false;
    for (const char* precision : {"fp32", "fp16", "bf16", "int8-bf16"}) {
        if (name == prefix + precision + ".full" || (allow_probe && name == prefix + precision + ".probe")) {
            p.precision = precision; recognized = true; break;
        }
    }
    if (!recognized) throw std::runtime_error("Engine is not a full, supported SS flow export (probes cannot generate assets)");
    if (weight_budget_mib >= 0) {
        if (p.engine->getStreamableWeightsSize() <= 0)
            throw std::runtime_error("SS weight budget requires an engine built with --weight-streaming");
        if (!p.engine->setWeightStreamingBudgetV2(static_cast<int64_t>(weight_budget_mib) << 20))
            throw std::runtime_error("TensorRT rejected the SS weight budget");
        fprintf(stderr, "[trellis] SS weight streaming: %d MiB budget, %lld MiB streamable, %lld MiB streaming scratch\n",
                weight_budget_mib, static_cast<long long>(p.engine->getStreamableWeightsSize() >> 20),
                static_cast<long long>(p.engine->getWeightStreamingScratchMemorySize() >> 20));
    }
    if (p.engine->getNbIOTensors() != 4) throw std::runtime_error("SS engine must have exactly four I/O tensors");
    const char* names[] = {"latent", "timestep", "cond", "velocity"};
    const std::vector<std::vector<int64_t>> shapes = {{1,4096,8}, {1}, {1,1029,1024}, {1,4096,8}};
    p.context.reset(p.engine->createExecutionContext());
    if (!p.context) throw std::runtime_error("TensorRT execution-context allocation failed");
    check_cuda(cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking));
    for (int i = 0; i < 4; ++i) {
        const auto dims = p.engine->getTensorShape(names[i]);
        if (dims.nbDims != static_cast<int>(shapes[i].size()) ||
            p.engine->getTensorDataType(names[i]) != nvinfer1::DataType::kFLOAT ||
            p.engine->getTensorLocation(names[i]) != nvinfer1::TensorLocation::kDEVICE ||
            p.engine->getTensorIOMode(names[i]) != (i == 3 ? nvinfer1::TensorIOMode::kOUTPUT : nvinfer1::TensorIOMode::kINPUT))
            throw std::runtime_error(std::string("Invalid SS engine I/O: ") + names[i]);
        size_t count = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] != shapes[i][j]) throw std::runtime_error(std::string("Invalid SS engine shape: ") + names[i]);
            count *= static_cast<size_t>(dims.d[j]);
        }
        p.buffers[i].allocate(count * sizeof(float));
        if (!p.context->setTensorAddress(names[i], p.buffers[i].data))
            throw std::runtime_error(std::string("Cannot bind TensorRT tensor: ") + names[i]);
    }
    fprintf(stderr, "[trellis] SS backend: TensorRT %d.%d.%d, context scratch %lld bytes (host transfers included)\n",
            NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
            static_cast<long long>(p.engine->getDeviceMemorySizeV2()));
}
TrtFlowRunner::~TrtFlowRunner() = default;
std::string TrtFlowRunner::precision() const { return impl_->precision; }

std::vector<float> TrtFlowRunner::forward(const std::vector<float>& latent, float timestep, const FlowCond& cond) {
    if (latent.size() != 8 * 4096 || !cond.cond || cond.proj || !std::isfinite(timestep))
        throw std::invalid_argument("Invalid TensorRT SS forward inputs");
    auto& p = *impl_;
    check_cuda(cudaSetDevice(p.device));
    check_cuda(cudaMemcpyAsync(p.buffers[0].data, latent.data(), latent.size() * sizeof(float), cudaMemcpyHostToDevice, p.stream));
    check_cuda(cudaMemcpyAsync(p.buffers[1].data, &timestep, sizeof(float), cudaMemcpyHostToDevice, p.stream));
    check_cuda(cudaMemcpyAsync(p.buffers[2].data, cond.cond, 1029 * 1024 * sizeof(float), cudaMemcpyHostToDevice, p.stream));
    if (!p.context->enqueueV3(p.stream)) throw std::runtime_error("TensorRT SS execution failed");
    std::vector<float> output(latent.size());
    check_cuda(cudaMemcpyAsync(output.data(), p.buffers[3].data, output.size() * sizeof(float), cudaMemcpyDeviceToHost, p.stream));
    check_cuda(cudaStreamSynchronize(p.stream));
    for (float value : output) if (!std::isfinite(value)) throw std::runtime_error("TensorRT SS produced NaN/Inf");
    return output;
}
}
#else
namespace trellis {
struct TrtFlowRunner::Impl {};
TrtFlowRunner::TrtFlowRunner(const std::string&, int, bool, int, bool) {
    throw std::runtime_error("TensorRT support is not built; configure with TRELLIS_TENSORRT=ON");
}
TrtFlowRunner::~TrtFlowRunner() = default;
std::string TrtFlowRunner::precision() const { return "unavailable"; }
std::vector<float> TrtFlowRunner::forward(const std::vector<float>&, float, const FlowCond&) {
    throw std::runtime_error("TensorRT support is not built");
}
}
#endif
