cmake_minimum_required(VERSION 3.18)

project(Assignment02 LANGUAGES CXX CUDA)

# Target the Tesla T4 architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Strict flags specifically for C++ files
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror;-O3>")

# Strict flags specifically for CUDA files (passing warnings down via -Xcompiler)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror;-O3;-lineinfo>")

add_subdirectory(task01)
add_subdirectory(matrix_io)
add_subdirectory(task02)
add_subdirectory(task03)
