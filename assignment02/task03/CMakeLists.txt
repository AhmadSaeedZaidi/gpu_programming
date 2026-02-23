add_executable(task03_app main.cpp gpu_wrapper.cu)

target_link_libraries(task03_app PRIVATE matrix_io_lib)
