import random
import sys

def generate_input_file(filename, M =3, K=4, N=2):
    with open(filename, 'w') as f:
        f.write(f"{M} {K}\n")
        for _ in range(M):
            row = [f"{random.uniform(-10.0, 10.0):.2f}" for _ in range(K)]
            f.write(" ".join(row) + "\n")

        f.write(f"{K} {N}\n")
        for _ in range(K):
            row = [f"{random.uniform(-10.0, 10.0):.2f}" for _ in range(N)]
            f.write(" ".join(row) + "\n")
            
    print(f"Successfully generated {filename} with A({M}x{K}) and B({K}x{N})")