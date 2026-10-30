cmake_minimum_required(VERSION 3.22)
project(aot_llama C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

# LLAMA_DIR được truyền từ workflow
add_subdirectory(${LLAMA_DIR} llama.cpp)

add_library(aot_llama SHARED aot_llama.cpp)
target_link_libraries(aot_llama PRIVATE llama ggml log android)
