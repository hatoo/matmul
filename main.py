import torch
from torch.utils.cpp_extension import load_inline
import time
import argparse
import os
import subprocess
from typing import Any

from collections.abc import Callable

with open("kernels/simple.cu", "r") as f:
    simple_src = f.read()

simple: Any = load_inline(
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


def dump_ptx(name: str, cuda_source: str, output_file: str | None = None):
    """
    Dump PTX code for a CUDA kernel.
    Args:
        name: kernel name
        cuda_source: CUDA source code
        output_file: output file path (optional, defaults to {name}.ptx)
    """
    if output_file is None:
        output_file = f"{name}.ptx"

    print(f"Dumping PTX for {name} kernel to {output_file}...")

    # Create a temporary source file
    temp_cu_file = f"temp_{name}.cu"
    with open(temp_cu_file, "w") as f:
        f.write(cuda_source)

    try:
        # Use nvcc to compile to PTX
        cmd = [
            "nvcc",
            "--ptx",
            "-O3",
            "-arch=sm_70",  # adjust based on your GPU architecture
            "-o",
            output_file,
            temp_cu_file,
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"PTX code successfully saved to {output_file}")
            # Display first few lines of PTX
            with open(output_file, "r") as f:
                lines = f.readlines()
                print(f"\nFirst 20 lines of {output_file}:")
                print("=" * 50)
                for i, line in enumerate(lines[:20]):
                    print(f"{i+1:3d}: {line.rstrip()}")
                if len(lines) > 20:
                    print(f"... ({len(lines) - 20} more lines)")
                print("=" * 50)
        else:
            print(f"Error generating PTX: {result.stderr}")

    finally:
        # Clean up temporary file
        if os.path.exists(temp_cu_file):
            os.remove(temp_cu_file)


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

    # Dump PTX mode
    ptx_parser = subparsers.add_parser(
        "dump-ptx", help="Dump PTX code for a selected kernel."
    )
    ptx_parser.add_argument(
        "kernel", choices=["simple"], help="Kernel to dump PTX for."
    )
    ptx_parser.add_argument(
        "-o", "--output", help="Output file for PTX code (default: {kernel}.ptx)."
    )

    # PTX dump mode
    subparsers.add_parser("dump_ptx", help="Dump PTX code for a kernel.")

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
    elif args.mode == "dump-ptx":
        if args.kernel == "simple":
            output_file = args.output if args.output else "simple.ptx"
            dump_ptx("simple", simple_src, output_file)
    elif args.mode == "verify":
        # Verify the kernel
        verify(n, m, k, "torch", launch_torch)
        verify(n, m, k, "simple", launch_simple)
    elif args.mode == "benchmark":
        # Benchmark the kernel
        benchmark(n, m, k, "torch", launch_torch)
        benchmark(n, m, k, "simple", launch_simple)
    elif args.mode == "dump_ptx":
        # Dump PTX code
        dump_ptx("simple", simple_src)


if __name__ == "__main__":
    main()
