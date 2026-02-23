
import matplotlib.pyplot as plt
import os
import subprocess
import sys

sys.path.append(os.path.abspath("."))
from matrix_io.generate_data import generate_input_file

dimensions = [
    (10, 10, 10),
    # gpu is now warmed up
    (512, 256, 512),
    (768, 512, 1024),
    (1024, 768, 1536),
    (1536, 2048, 1024),
    (2048, 1024, 3072),
    (3072, 4096, 2048),
    (4096, 2048, 5120),
]

cpu_times = []
gpu_times = []
labels = []

for M, K, N in dimensions:
    label = f"({M}x{K}) * ({K}x{N})"
    labels.append(label)

    print(f"\n--- Benchmarking Dimensions: {label} ---")
    generate_input_file("bench_input.txt", M, K, N)

    # Run CPU executable
    cpu_result = subprocess.run(["./build/task02/task02_app", "bench_input.txt", "output_cpu.txt"], capture_output=True, text=True)
    try:
        cpu_time = float(cpu_result.stdout.strip())
        cpu_times.append(cpu_time)
    except ValueError:
        print("Error parsing CPU time. Output was:", cpu_result.stdout)
        cpu_times.append(0)

    # Run GPU executable
    gpu_result = subprocess.run(["./build/task03/task03_app", "bench_input.txt", "output_gpu.txt"], capture_output=True, text=True)
    try:
        gpu_time = float(gpu_result.stdout.strip())
        gpu_times.append(gpu_time)
    except ValueError:
        print("Error parsing GPU time. Output was:", gpu_result.stdout)
        gpu_times.append(0)


# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(labels, cpu_times, label='CPU Time (ms)', marker='o', color='red')
plt.plot(labels, gpu_times, label='GPU Time (ms) [inc. Memory Transfer]', marker='s', color='blue')
plt.title("Matrix Multiplication: CPU vs GPU (Non-Square Matrices)")
plt.xlabel("Matrix Dimensions (M x K * K x N)")
plt.ylabel("Execution Time (milliseconds)")
plt.xticks(rotation=60)
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.savefig("task04_benchmark_plot.png")
print("\nBenchmarking complete! Plot saved as 'task04_benchmark_plot.png'")