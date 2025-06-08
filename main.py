import torch
from torch.utils.cpp_extension import load_inline
import time
import argparse

from collections.abc import Callable

with open('kernels/simple.cu', 'r') as f:
    simple_src = f.read()

simple = load_inline(
    name='simple',
    cpp_sources='void simple(uintptr_t a, uintptr_t b, uintptr_t c, int n, int m, int k);',
    cuda_sources=simple_src,
    functions='simple',
    with_cuda=True,
    extra_cuda_cflags=['-O3'],
)

def launch_simple(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    n, k = a.shape
    _k, m = b.shape
    
    # Call the CUDA kernel
    simple.simple(a.data_ptr(), b.data_ptr(), c.data_ptr(), n, m, k)


def benchmark(
    n: int,
    m: int,
    k: int,
    name: str,
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]):
    a = torch.randn(n, k, dtype=torch.float32, device='cuda')
    b = torch.randn(k, m, dtype=torch.float32, device='cuda')
    c = torch.zeros(n, m, dtype=torch.float32, device='cuda')

    warmup_times = 4
    times = 8
    sum = 0

    for _ in range(warmup_times):
        torch.cuda.synchronize()
        kernel(a, b, c)
        torch.cuda.synchronize()

    for _ in range(times):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        kernel(a, b, c)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        sum += end - start
    average_time = sum / times

    print(f"{name} kernel: {n}x{m}x{k} took {average_time / 1e6:.2f} ms on average.")

def verify(
    n: int,
    m: int,
    k: int,
    name: str,
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]):
    a = torch.randn(n, k, dtype=torch.float32, device='cuda')
    b = torch.randn(k, m, dtype=torch.float32, device='cuda')
    c = torch.zeros(n, m, dtype=torch.float32, device='cuda')
    c_ref = torch.zeros(n, m, dtype=torch.float32, device='cuda')
    c_ref = torch.matmul(a, b)

    kernel(a, b, c)
    torch.cuda.synchronize()
    if torch.allclose(c, c_ref, rtol=1e-02, atol=1e-03):
        print(f"{name} kernel verification passed.")
    else:
        print(f"{name} kernel verification failed.")

def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA kernels.")
    parser.add_argument('--verify', action='store_true', help='Verify the kernel correctness.')

    args = parser.parse_args()

    n, m, k = 4096, 4096, 4096

    # Define a simple kernel function for matrix multiplication
    def launch_torch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        torch.matmul(a, b, out=c)

    if args.verify:
        # Verify the kernel
        verify(n, m, k, 'torch', launch_torch)
        verify(n, m, k, 'simple', launch_simple)
    else:
        # Benchmark the kernel
        benchmark(n, m, k, 'torch', launch_torch)
        benchmark(n, m, k, 'simple', launch_simple)


if __name__ == "__main__":
    main()
