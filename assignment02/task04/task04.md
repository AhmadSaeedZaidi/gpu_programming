# observations!
## 1. Small Matrices
### Observation:
For exceptionally small workloads (e.g., the 10x10 warmup matrix), the CPU vastly outperforms the GPU.

### Why it happens: 
The GPU requires us to allocate device memory (cudaMalloc) and send the data across the PCIe bus (cudaMemcpy). For a 10x10 matrix, the CPU can finish the actual math in microseconds—much faster than the time it takes the GPU just to receive the data.

## 2.GPU Graph is a Flatline

### Observation: 
Even when intentionally bottlenecking the GPU with an extreme inner dimension (e.g., (128x38000) * (38000x128)), the GPU execution time remains completely flat at roughly 400ms.

### Why it happens: 
Even at this scale (roughly 1 to 2 billion operations), the GPU's math cores are so incredibly fast, and the simple 128x128 paralellism helps so much, that the computation takes mere milliseconds. The ~400ms we are seeing is almost entirely the fixed cost of memory allocation and data transfer. The GPU is still mostly waiting on memory.

## 3. CPU Graph Exponential growth

### Observation:
CPU line acts erratically, directly reflecting the total floating-point operations (FLOPs).

### Why it happens: 
Massive spike at (1024x768) * (768x1536). This configuration hits roughly 2.4 billion sequential operations. Because the CPU has to process these one by one, its execution time skyrockets to over 3000ms. The CPU is purely compute-bound, and changes in total operations causes massive spikes.