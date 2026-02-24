import matplotlib.pyplot as plt
import os
import subprocess
import sys

sys.path.append(os.path.abspath("."))
from matrix_io.generate_data import generate_input_file

# --- CONFIGURATION ---
small_dimensions = [
    (10, 10, 10),           # Warmup GPU
    (512, 256, 512),        # ~134 Million ops
    (768, 512, 1024),       # ~805 Million ops
    (1024, 768, 1536),      # ~2.4 Billion ops
    (1536, 1024, 1536),     # ~4.8 Billion ops
    (2048, 1024, 2048),     # ~8.5 Billion ops
    (2048, 2048, 2048)      # ~17 Billion ops
]
large_dimensions = [
    (2048, 2048, 2048),     # ~17 Billion ops
    (3072, 3072, 3072),     # ~57 Billion ops
    (4096, 4096, 4096),     # ~137 Billion ops
    (6144, 4096, 6144),     # ~309 Billion ops
    (8192, 8192, 8192)      # ~1.1 Trillion ops
]

def run_benchmark(executable_args):
    result = subprocess.run(executable_args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {' '.join(executable_args)}:\n{result.stderr.strip()}")
        return 0.0
    try:
        return float(result.stdout.strip().split('\n')[-1])
    except ValueError:
        print(f"Error parsing time. Output was: {result.stdout}")
        return 0.0

print("=======================================")
print("PHASE 1: CPU vs GPU vs GPU (Tiled)")
print("=======================================")

labels_small, cpu_times, gpu_naive_times_small, gpu_tiled_times_small = [], [], [], []

for M, K, N in small_dimensions:
    label = f"{M}x{K}x{N}"
    labels_small.append(label)
    print(f"\n--- Dimensions: {label} ---")
    generate_input_file("bench_input.txt", M, K, N)

    # 1. CPU
    cpu_times.append(run_benchmark(["./build/task02/task02_app", "bench_input.txt", "output_cpu.txt"]))
    # 2. GPU Naive
    gpu_naive_times_small.append(run_benchmark(["./build/task03/task03_app", "bench_input.txt", "output_gpu.txt"]))
    # 3. GPU Tiled
    gpu_tiled_times_small.append(run_benchmark(["./build/task03/task03_app", "bench_input.txt", "output_gpu.txt", "--tiled"]))

# Plot Phase 1

savefile = "results/gpu_v_cpu.png"
plt.figure(figsize=(10, 6))
plt.plot(labels_small[1:], cpu_times[1:], label='CPU Time', marker='o', color='red') # Skip warmup index 0
plt.plot(labels_small[1:], gpu_naive_times_small[1:], label='GPU (Naive)', marker='s', color='blue')
plt.plot(labels_small[1:], gpu_tiled_times_small[1:], label='GPU (Tiled Shared Mem)', marker='^', color='green')
plt.title("Phase 1: CPU vs GPU Performance (Up to ~8.5B Ops)")
plt.xlabel("Matrix Dimensions (M x K x N)")
plt.ylabel("Execution Time (ms)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.savefig(savefile)
print(f"\nPhase 1 Plot saved as '{savefile}'")


print("\n=======================================")
print("PHASE 2: GPU Naive vs GPU (Tiled)")
print("=======================================")

labels_large, gpu_naive_times_large, gpu_tiled_times_large = [], [], []

for M, K, N in large_dimensions:
    label = f"{M}x{K}x{N}"
    labels_large.append(label)
    print(f"\n--- Dimensions: {label} ---")
    generate_input_file("bench_input.txt", M, K, N)

    # 1. GPU Naive
    gpu_naive_times_large.append(run_benchmark(["./build/task03/task03_app", "bench_input.txt", "output_gpu.txt"]))
    # 2. GPU Tiled
    gpu_tiled_times_large.append(run_benchmark(["./build/task03/task03_app", "bench_input.txt", "output_gpu.txt", "--tiled"]))

# Plot Phase 2
savefile = "results/tiled_v_naive.png"

plt.figure(figsize=(10, 6))
plt.plot(labels_large, gpu_naive_times_large, label='GPU (Naive Global Mem)', marker='s', color='blue')
plt.plot(labels_large, gpu_tiled_times_large, label='GPU (Tiled Shared Mem)', marker='^', color='green')
plt.title("Phase 2: GPU Naive vs Tiled Performance (Up to 1.1 Trillion Ops)")
plt.xlabel("Matrix Dimensions (M x K x N)")
plt.ylabel("Execution Time (ms)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.legend()
plt.grid(True)

plt.savefig(savefile)
print(f"\nPhase 2 Plot saved as '{savefile}'")