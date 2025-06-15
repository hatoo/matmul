import torch
from torch.utils.cpp_extension import load_inline
import time
import argparse

from collections.abc import Callable

with open("kernels/simple.cu", "r") as f:
    simple_src = f.read()

simple = load_inline(
    name="simple",
    cpp_sources="void simple(uintptr_t a, uintptr_t b, uintptr_t c, int n, int m, int k);",
    cuda_sources=simple_src,
    functions="simple",
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
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
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
    warmup_times: int = 4,
    times: int = 8,
):
    """
    Benchmark a kernel for matrix multiplication.
    Args:
        n, m, k: matrix dimensions
        name: kernel name
        kernel: function to run (a, b, c)
        warmup_times: number of warmup runs
        times: number of timed runs
    """
    a = torch.randn(n, k, dtype=torch.float32, device="cuda")
    b = torch.randn(k, m, dtype=torch.float32, device="cuda")
    c = torch.zeros(n, m, dtype=torch.float32, device="cuda")

    # Warmup
    for _ in range(warmup_times):
        torch.cuda.synchronize()
        kernel(a, b, c)
        torch.cuda.synchronize()

    # Timed runs
    elapsed_times = []
    for _ in range(times):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        kernel(a, b, c)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        elapsed_times.append(end - start)
    average_time = sum(elapsed_times) / times
    min_time = min(elapsed_times)
    max_time = max(elapsed_times)

    print(
        f"{name} kernel: {n}x{m}x{k} avg: {average_time / 1e6:.2f} ms, min: {min_time / 1e6:.2f} ms, max: {max_time / 1e6:.2f} ms."
    )


def verify(
    n: int,
    m: int,
    k: int,
    name: str,
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
):
    a = torch.randn(n, k, dtype=torch.float32, device="cuda")
    b = torch.randn(k, m, dtype=torch.float32, device="cuda")
    c = torch.zeros(n, m, dtype=torch.float32, device="cuda")
    c_ref = torch.zeros(n, m, dtype=torch.float32, device="cuda")
    c_ref = torch.matmul(a, b)

    kernel(a, b, c)
    torch.cuda.synchronize()
    if torch.allclose(c, c_ref, rtol=1e-02, atol=1e-03):
        print(f"{name} kernel verification passed.")
    else:
        print(f"{name} kernel verification failed.")


def profile(
    n: int,
    m: int,
    k: int,
    name: str,
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
):
    a = torch.randn(n, k, dtype=torch.float32, device="cuda")
    b = torch.randn(k, m, dtype=torch.float32, device="cuda")
    c = torch.zeros(n, m, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    kernel(a, b, c)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()
    elapsed = (end - start) / 1e6
    print(
        f"{name} kernel: {n}x{m}x{k} took {elapsed:.2f} ms (profile mode, single run)."
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA kernels.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Benchmark mode
    subparsers.add_parser("benchmark", help="Benchmark all kernels.")

    # Verify mode
    subparsers.add_parser("verify", help="Verify all kernels.")

    # Profile mode
    profile_parser = subparsers.add_parser(
        "profile", help="Profile a selected kernel (single run)."
    )
    profile_parser.add_argument(
        "kernel", choices=["torch", "simple"], help="Kernel to profile."
    )

    args = parser.parse_args()

    n, m, k = 4096, 4096, 4096

    # Define a simple kernel function for matrix multiplication
    def launch_torch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        torch.matmul(a, b, out=c)

    if args.mode == "profile":
        if args.kernel == "torch":
            profile(n, m, k, "torch", launch_torch)
        elif args.kernel == "simple":
            profile(n, m, k, "simple", launch_simple)
    elif args.mode == "verify":
        # Verify the kernel
        verify(n, m, k, "torch", launch_torch)
        verify(n, m, k, "simple", launch_simple)
    elif args.mode == "benchmark":
        # Benchmark the kernel
        benchmark(n, m, k, "torch", launch_torch)
        benchmark(n, m, k, "simple", launch_simple)


if __name__ == "__main__":
    main()
