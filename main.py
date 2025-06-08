import torch
from torch.utils.cpp_extension import load_inline
import time

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

    print(f"Time taken for {n}x{m} and {m}x{k} matrix multiplication: {average_time / 1e6:.2f} ms")

def main():
    n, m, k = 8192, 8192, 8192

    # Define a simple kernel function for matrix multiplication
    def kernel(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        torch.matmul(a, b, out=c)

    # Benchmark the kernel
    benchmark(n, m, k, kernel)
    benchmark(n, m, k, launch_simple)


if __name__ == "__main__":
    main()
