#include <stdio.h>

int main() {
    int deviceId = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, deviceId);

    // Fetch clocks using attributes (Required for CUDA 13.0+)
    int clockRateKHz = 0;
    int memoryClockRateKHz = 0;
    cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, deviceId);
    cudaDeviceGetAttribute(&memoryClockRateKHz, cudaDevAttrMemoryClockRate, deviceId);
    
    printf("--- 8 Important Properties (from cudaDeviceProp) ---\n");
    
    // 1. Device Name 
    printf("devProp.name: represents the GPU Model. Value = %s\n", devProp.name);
    
    // 2. Global Memory 
    printf("devProp.totalGlobalMem: represents the total VRAM in the GPU. Value = %zu bytes\n", devProp.totalGlobalMem);
    
    // 3. Streaming Multiprocessors (SMs)
    printf("devProp.multiProcessorCount: represents how many SMs (fundamental units of processing) are in the GPU. Value = %d\n", devProp.multiProcessorCount);
    
    // 4. Max Threads per Block
    printf("devProp.maxThreadsPerBlock: represents the maximum number of threads that can be launched in a single block. Value = %d\n", devProp.maxThreadsPerBlock);
    
    // 5. Warp Size
    printf("devProp.warpSize: represents the number of threads (32) grouped by the SM to execute instructions in lockstep. Value = %d\n", devProp.warpSize);
    
    // 6. Shared Memory per Block
    printf("devProp.sharedMemPerBlock: represents how much fast on-chip cache/SRAM is available per block. Value = %zu bytes\n", devProp.sharedMemPerBlock);
    
    // 7. Max Grid Size
    printf("devProp.maxGridSize[0]: represents the maximum number of blocks allowed in the X direction. Value = %d\n", devProp.maxGridSize[0]);
    
    // 8. Memory Bus Width
    printf("devProp.memoryBusWidth: represents how many bits can pass simultaneously across the memory bus. Value = %d bits\n", devProp.memoryBusWidth);
    

    printf("\n---Calculated Values---\n");

    // 9. Max Global Memory Bandwidth (GB/s)
    double memBandwidth = 2.0 * memoryClockRateKHz * (devProp.memoryBusWidth / 8.0) / 1000000.0;
    printf("Max Global Memory Bandwidth: represents how fast memory can be accessed in the global memory. Value = %.2f GB/s\n", memBandwidth);

    // 10. Peak Compute Performance (TFLOPs)
    int coresPerSM = 64; // Hardcoded for Tesla T4 (64 cores per SM)
    double peakTFLOPs = (2.0 * devProp.multiProcessorCount * coresPerSM * clockRateKHz) / 1e9;
    printf("Peak Compute Performance: represents how many trillions of operations can be performed by the GPU/s in ideal conditions. Value = %.2f TFLOPs\n", peakTFLOPs);

    
    

    return 0;
}
