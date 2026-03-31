cmake_minimum_required(VERSION 3.18)
project(CudaDenoise LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target RTX A2000 (Ampere architecture, Compute Capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Define the executable and its source files
add_executable(cuda-denoise
    main.cpp
    ppm_utils.cpp
    denoise_cpu.cpp
    denoise_gpu.cu
)

# Apply strict warnings and optimizations
# CXX flags for host code, CUDA flags with -Xcompiler to pass warnings to the host compiler
target_compile_options(cuda-denoise PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra -O3>
)