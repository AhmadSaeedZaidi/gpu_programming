#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "hpc.h"
#include "ppm_utils.h"
#include "denoise.h"

int main(void) {
    cudaFree(0);

    PPM_image img_cpu, img_gpu;
    read_ppm(stdin, &img_cpu);
    
    img_gpu.width = img_cpu.width;
    img_gpu.height = img_cpu.height;
    img_gpu.maxcol = img_cpu.maxcol;
    img_gpu.r = (unsigned char*)malloc(img_cpu.width * img_cpu.height);
    img_gpu.g = (unsigned char*)malloc(img_cpu.width * img_cpu.height);
    img_gpu.b = (unsigned char*)malloc(img_cpu.width * img_cpu.height);
    memcpy(img_gpu.r, img_cpu.r, img_cpu.width * img_cpu.height);
    memcpy(img_gpu.g, img_cpu.g, img_cpu.width * img_cpu.height);
    memcpy(img_gpu.b, img_cpu.b, img_cpu.width * img_cpu.height);

    double tstart = hpc_gettime();
    denoise_cpu(img_cpu.r, img_cpu.width, img_cpu.height);
    denoise_cpu(img_cpu.g, img_cpu.width, img_cpu.height);
    denoise_cpu(img_cpu.b, img_cpu.width, img_cpu.height);
    double elapsed_cpu = hpc_gettime() - tstart;
    fprintf(stderr, "CPU Execution time: %.3f s\n", elapsed_cpu);

    tstart = hpc_gettime();
    denoise_gpu(img_gpu.r, img_gpu.g, img_gpu.b, img_gpu.width, img_gpu.height);
    double elapsed_gpu = hpc_gettime() - tstart;
    fprintf(stderr, "GPU Execution time: %.3f s\n", elapsed_gpu);

    write_ppm(stdout, &img_gpu, "produced by cuda-denoise.cu (GPU)");

    free_ppm(&img_cpu);
    free_ppm(&img_gpu);
    return EXIT_SUCCESS;
}