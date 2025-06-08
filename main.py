import torch
import time

from collections.abc import Callable

def benchmark(
    n: int,
    m: int,
    k: int,
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]):
    a = torch.randn(n, m, device='cuda')
    b = torch.randn(m, k, device='cuda')
    c = torch.zeros(n, k, device='cuda')

    warmup_times = 4
    times = 8
    sum = 0

    for _ in range(warmup_times):
        torch.cuda.synchronize()
        kernel(a, b, c)

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


if __name__ == "__main__":
    main()
