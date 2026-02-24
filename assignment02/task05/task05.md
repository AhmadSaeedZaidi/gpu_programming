# Phase 2 Observations: Naive vs Tiled GPU (and Hardware Quirks)

## 1. Tiling performs best with High K
### Observation:
When the inner dimension K is exceptionally large, the Tiled GPU implementation significantly outperforms the Naive global memory approach.

### Why it happens:
The K dimension dictates how many times the inner loop runs. With Tiling, threads load a small 16x16 block of the matrix into ultra-fast Shared Memory and reuse it. When K is huge, that same fast memory block is reused hundreds or thousands of times. High K means high data reuse, which is exactly what Tiling is built to exploit. 

## 2. Square-ish Matrices Provide Increasing Gains
### Observation:
As matrices scale up in a balanced, roughly square manner, the performance gap between the Naive and Tiled kernels steadily widens, and GPU generally performs better.

### Why it happens:
Square matrices allow for near-perfect memory coalescing. When the grid is balanced, threads can pull contiguous chunks of memory in a highly predictable pattern. As the total size grows, the memory bus gets saturated. Tiling alleviates this traffic jam by cutting down global memory trips, making the memory savings much more noticeable at larger scales.

## 3. The FP32 vs FP64 Hardware Reality
### Observation:
Switching the data type from `double` (FP64) to `float` (FP32) resulted in roughly double the performance on the Colab Tesla T4 GPU.

### Why it happens:
The T4 is built for machine learning, meaning it has an overwhelming number of FP32 cores but very few FP64 cores. By switching to `float`, we not only halved our memory bandwidth requirements (4 bytes instead of 8 bytes per element), but we also unlocked the GPU's primary compute cores, resulting in a massive speedup.

## 4. Square Outperforms Extreme Rectangular
### Observation:
Even when the total number of math operations (FLOPs) is identical, square matrices execute noticeably faster than extremely skewed rectangular matrices.

### Why it happens:
Extreme aspect ratios (like a very thin and tall grid) can lead to under-utilized thread blocks. For example, boundary threads might sit idle if the dimensions don't divide cleanly by 32.

## 5. Thread Block Occupancy and Latency Hiding

### Observation:
Shrinking the tile size from 32x32 (1024 threads) to 16x16 (256 threads) improved overall performance.

### Why it happens:
A massive 1024-thread block monopolizes a Streaming Multiprocessor's (SM) registers, leaving no room to schedule other blocks. By using smaller 256-thread blocks, the SM can load multiple blocks at once (higher "occupancy").

## 6. Padding Trick

### Observation:
Allocating the arrays as `[16][17]` (adding a +1 padding to the columns) yielded a minor but measurable performance improvement over a standard `[16][16]`.

### Why it happens:
GPU Shared Memory is divided into 32 distinct memory modules called "banks". If multiple threads in a warp (a group of 32 threads) try to read from the *same* memory bank at the same time, the hardware forces them to wait in line. This is called a "bank conflict" and it severely slows down memory reads. 

During matrix multiplication, our threads need to read down the *columns* of the shared memory tile for Matrix B. If our tile is `[16][16]`, jumping down a column means jumping exactly 16 memory addresses at a time. Because 16 is a clean multiple of 32, many threads end up requesting data from the exact same memory bank simultaneously. 

By padding the array to `[16][17]`, we shift the memory alignment. Jumping down a column now requires jumping 17 addresses. Since 17 is not a multiple of 32, the memory requests are naturally staggered across different, independent memory banks, allowing all threads to grab their data in parallel.

