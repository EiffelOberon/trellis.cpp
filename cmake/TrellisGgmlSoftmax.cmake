# Keep the pinned submodule clean. Generate a narrowly patched softmax source
# for the SDK build; fail on upstream drift instead of silently skipping it.
# GGML requests the device-wide maximum opt-in shared-memory size even for
# launches fitting the default limit. That request can fail when sharing a
# primary context with OptiX. Set only the launch's required limit, when needed.
set(_softmax_source "${GGML_DIR}/src/ggml-cuda/softmax.cu")
file(READ "${_softmax_source}" _softmax)
foreach(_specialization "true, ncols, block, T" "true, 0, 0, T")
  set(_before "CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<${_specialization}>), smpbo);")
  string(FIND "${_softmax}" "${_before}" _found)
  if(_found LESS 0)
    message(FATAL_ERROR "Pinned GGML softmax source changed; review the editor shared-memory patch")
  endif()
  string(REPLACE "${_before}"
    "if (nbytes_shared > ggml_cuda_info().devices[id].smpb) { CUDA_CHECK(cudaFuncSetAttribute((soft_max_f32<${_specialization}>), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared)); }"
    _softmax "${_softmax}")
endforeach()
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/softmax_editor.cu" CONTENT "${_softmax}" @ONLY)
get_target_property(_ggml_cuda_sources ggml-cuda SOURCES)
list(FILTER _ggml_cuda_sources EXCLUDE REGEX "(^|/)softmax\\.cu$")
set_property(TARGET ggml-cuda PROPERTY SOURCES "${_ggml_cuda_sources}")
target_sources(ggml-cuda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated/softmax_editor.cu")
target_include_directories(ggml-cuda PRIVATE "${GGML_DIR}/src/ggml-cuda")
