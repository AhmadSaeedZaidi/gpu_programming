#include "gpu_wrapper.hpp"
#include <iostream>
#include <stdexcept>

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

Matrix multiplyMatricesGPU(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions do not match!");
    }

    int M = A.rows;
    int K = A.cols;
    int N = B.cols;

    Matrix C;
    C.rows = M;
    C.cols = N;
    C.data.resize(M * N, 0.0);

    double *d_A, *d_B, *d_C;
    size_t bytesA = M * K * sizeof(double);
    size_t bytesB = K * N * sizeof(double);
    size_t bytesC = M * N * sizeof(double);

    cudaCheckError(cudaMalloc(&d_A, bytesA));
    cudaCheckError(cudaMalloc(&d_B, bytesB));
    cudaCheckError(cudaMalloc(&d_C, bytesC));

    cudaCheckError(cudaMemcpy(d_A, A.data.data(), bytesA, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, B.data.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    
    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(C.data.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}
