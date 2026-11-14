cmake_minimum_required(VERSION 3.18)
project(trellis_glint_import LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(GLINT_SOURCE_DIR "" CACHE PATH "Existing Glint checkout")
set(GLINT_BUILD_DIR "" CACHE PATH "Compatible built Glint libraries")
find_library(GLINT_IO_LIBRARY NAMES glint_io HINTS "${GLINT_BUILD_DIR}/lib" REQUIRED NO_DEFAULT_PATH)
find_library(GLINT_SCENE_LIBRARY NAMES glint_scene HINTS "${GLINT_BUILD_DIR}/lib" REQUIRED NO_DEFAULT_PATH)
add_executable(trellis-check-glint check_import.cpp)
target_include_directories(trellis-check-glint PRIVATE
    "${GLINT_SOURCE_DIR}/third_party/glint_io/include"
    "${GLINT_SOURCE_DIR}/third_party/glint_scene/include"
    "${GLINT_BUILD_DIR}/_deps/glm-src")
target_link_libraries(trellis-check-glint PRIVATE "${GLINT_IO_LIBRARY}" "${GLINT_SCENE_LIBRARY}")
target_compile_definitions(trellis-check-glint PRIVATE GLM_FORCE_DEPTH_ZERO_TO_ONE NOMINMAX)
