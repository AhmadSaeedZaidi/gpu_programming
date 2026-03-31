#include "denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define cudaSafeCall(err) \
    do { \
        cudaError_t err_val = (err); \
        if (err_val != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                    __FILE__, __LINE__, err_val, cudaGetErrorString(err_val)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define cudaCheckError() \
    do { \
        cudaError_t err_val = cudaGetLastError(); \
        if (err_val != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                    __FILE__, __LINE__, err_val, cudaGetErrorString(err_val)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define TILE_W 32
#define TILE_H 32
#define BLOCK_THREADS 256

__device__ __forceinline__ void compare_and_swap_dev(unsigned char *a, unsigned char *b) {
    unsigned char val_a = *a;
    unsigned char val_b = *b;
    *a = (val_a < val_b) ? val_a : val_b;
    *b = (val_a > val_b) ? val_a : val_b;
}

__device__ __forceinline__ unsigned char median_of_five_dev(unsigned char v[5]) {
    compare_and_swap_dev(v + 3, v + 4);
    compare_and_swap_dev(v + 2, v + 3);
    compare_and_swap_dev(v + 1, v + 2);
    compare_and_swap_dev(v, v + 1);
    compare_and_swap_dev(v + 3, v + 4);
    compare_and_swap_dev(v + 2, v + 3);
    compare_and_swap_dev(v + 1, v + 2);
    compare_and_swap_dev(v + 3, v + 4);
    compare_and_swap_dev(v + 2, v + 3);
    return v[2];
}

__global__ void denoise_kernel_3d_optimized(
    const unsigned char * __restrict__ r_in, const unsigned char * __restrict__ g_in, const unsigned char * __restrict__ b_in,
    unsigned char * __restrict__ r_out, unsigned char * __restrict__ g_out, unsigned char * __restrict__ b_out,
    int width, int height) 
{
    int ch = blockIdx.z; 
    
    const unsigned char *d_in = (ch == 0) ? r_in : ((ch == 1) ? g_in : b_in);
    unsigned char *d_out      = (ch == 0) ? r_out : ((ch == 1) ? g_out : b_out);

    __shared__ unsigned char tile[TILE_H + 2][TILE_W + 2];

    int tid = threadIdx.x;
    int start_x = blockIdx.x * TILE_W - 1;
    int start_y = blockIdx.y * TILE_H - 1;
    int tile_elements = (TILE_W + 2) * (TILE_H + 2);

    for (int i = tid; i < tile_elements; i += BLOCK_THREADS) {
        int lx = i % (TILE_W + 2);
        int ly = i / (TILE_W + 2);
        int gx = start_x + lx;
        int gy = start_y + ly;

        int clamp_x = max(0, min(width - 1, gx));
        int clamp_y = max(0, min(height - 1, gy));

        tile[ly][lx] = d_in[clamp_y * width + clamp_x];
    }

    __syncthreads();

    int compute_elements = TILE_W * TILE_H;
    for (int i = tid; i < compute_elements; i += BLOCK_THREADS) {
        int lx = i % TILE_W;
        int ly = i / TILE_W;
        int gx = blockIdx.x * TILE_W + lx;
        int gy = blockIdx.y * TILE_H + ly;

        if (gx < width && gy < height) {
            int sx = lx + 1;
            int sy = ly + 1;

            unsigned char v[5];
            v[0] = tile[sy][sx];
            v[1] = tile[sy][sx - 1];
            v[2] = tile[sy][sx + 1];
            v[3] = tile[sy - 1][sx];
            v[4] = tile[sy + 1][sx];
            
            d_out[gy * width + gx] = median_of_five_dev(v);
        }
    }
}

void denoise_gpu(unsigned char *r, unsigned char *g, unsigned char *b, int width, int height) {
    size_t bytes = (size_t)width * height * sizeof(unsigned char);
    
    unsigned char *d_in, *d_out;
    cudaSafeCall(cudaMalloc((void**)&d_in, 3 * bytes));
    cudaSafeCall(cudaMalloc((void**)&d_out, 3 * bytes));

    unsigned char *d_r_in = d_in;
    unsigned char *d_g_in = d_in + bytes;
    unsigned char *d_b_in = d_in + 2 * bytes;

    unsigned char *d_r_out = d_out;
    unsigned char *d_g_out = d_out + bytes;
    unsigned char *d_b_out = d_out + 2 * bytes;

    cudaSafeCall(cudaMemcpy(d_r_in, r, bytes, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_g_in, g, bytes, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_b_in, b, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_THREADS); 
    dim3 grid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H, 3);

    denoise_kernel_3d_optimized<<<grid, block>>>(d_r_in, d_g_in, d_b_in, d_r_out, d_g_out, d_b_out, width, height);
    cudaCheckError();

    cudaSafeCall(cudaMemcpy(r, d_r_out, bytes, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(g, d_g_out, bytes, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(b, d_b_out, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
}