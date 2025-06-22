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

with open("kernels/tile.cu", "r") as f:
    tile_src = f.read()

simple: Any = load_inline(
    name="simple",
    cpp_sources="void simple(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k);",
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


tile: Any = load_inline(
    name="tile",
    cpp_sources="void tile(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k);",
    cuda_sources=tile_src,
    functions="tile",
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
)


def launch_tile(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    n, k = a.shape
    _k, m = b.shape

    # Call the CUDA kernel
    tile.tile(a.data_ptr(), b.data_ptr(), c.data_ptr(), n, m, k)


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
    m: int,
    n: int,
    k: int,
    name: str,
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
):
    a = torch.randn(m, k, dtype=torch.float32, device="cuda")
    b = torch.randn(k, n, dtype=torch.float32, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    c_ref = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    c_ref = torch.matmul(a, b)

    kernel(a, b, c)
    torch.cuda.synchronize()
    if torch.allclose(c, c_ref, rtol=1e-02, atol=1e-03):
        print(f"{name} kernel verification passed.")
    else:
        print(f"{name} kernel verification failed.")


def profile(
    m: int,
    n: int,
    k: int,
    name: str,
    kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
):
    a = torch.randn(m, k, dtype=torch.float32, device="cuda")
    b = torch.randn(k, n, dtype=torch.float32, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    kernel(a, b, c)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()
    elapsed = (end - start) / 1e6
    print(
        f"{name} kernel: {n}x{m}x{k} took {elapsed:.2f} ms (profile mode, single run)."
    )


def dump_ptx(
    name: str, cuda_source: str, output_file: str | None = None, debug: bool = False
):
    """
    Dump PTX code for a CUDA kernel.
    Args:
        name: kernel name
        cuda_source: CUDA source code
        output_file: output file path (optional, defaults to {name}.ptx)
        debug: include debug information (default: False)
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
            "-O3" if not debug else "-O0",
            "-arch=sm_120",  # adjust based on your GPU architecture
            "-o",
            output_file,
            temp_cu_file,
        ]

        if debug:
            cmd.extend(["-g", "-G", "--source-in-ptx"])

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


def dump_sass(
    name: str, cuda_source: str, output_file: str | None = None, debug: bool = False
):
    """
    Dump SASS (CUDA assembly) code for a CUDA kernel.
    Args:
        name: kernel name
        cuda_source: CUDA source code
        output_file: output file path (optional, defaults to {name}.sass)
        debug: include debug information (default: False)
    """
    if output_file is None:
        output_file = f"{name}.sass"

    print(f"Dumping SASS for {name} kernel to {output_file}...")

    # Create a temporary source file
    temp_cu_file = f"temp_{name}.cu"
    temp_ptx_file = f"temp_{name}.ptx"
    temp_cubin_file = f"temp_{name}.cubin"

    with open(temp_cu_file, "w") as f:
        f.write(cuda_source)

    try:
        # Step 1: Compile to cubin
        cmd_cubin = [
            "nvcc",
            "-cubin",
            "-O3" if not debug else "-O0",
            "-arch=sm_120",  # adjust based on your GPU architecture
            "-o",
            temp_cubin_file,
            temp_cu_file,
        ]

        if debug:
            cmd_cubin.extend(["-g", "-G"])

        print(f"Running: {' '.join(cmd_cubin)}")
        result = subprocess.run(cmd_cubin, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error generating cubin: {result.stderr}")
            return

        # Step 2: Extract SASS from cubin using cuobjdump
        cmd_sass = [
            "cuobjdump",
            "--dump-sass",
            temp_cubin_file,
        ]

        print(f"Running: {' '.join(cmd_sass)}")
        result = subprocess.run(cmd_sass, capture_output=True, text=True)

        if result.returncode == 0:
            # Write SASS output to file
            with open(output_file, "w") as f:
                f.write(result.stdout)

            print(f"SASS code successfully saved to {output_file}")
            # Display first few lines of SASS
            lines = result.stdout.split("\n")
            print(f"\nFirst 30 lines of {output_file}:")
            print("=" * 50)
            for i, line in enumerate(lines[:30]):
                print(f"{i+1:3d}: {line}")
            if len(lines) > 30:
                print(f"... ({len(lines) - 30} more lines)")
            print("=" * 50)
        else:
            print(f"Error generating SASS: {result.stderr}")

    finally:
        # Clean up temporary files
        for temp_file in [temp_cu_file, temp_ptx_file, temp_cubin_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)


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

    # Dump mode (PTX or SASS)
    dump_parser = subparsers.add_parser(
        "dump", help="Dump PTX or SASS code for a selected kernel."
    )
    dump_parser.add_argument(
        "kernel", choices=["simple"], help="Kernel to dump code for."
    )
    dump_parser.add_argument(
        "-f",
        "--format",
        choices=["ptx", "sass"],
        default="ptx",
        help="Output format: PTX or SASS (default: ptx).",
    )
    dump_parser.add_argument(
        "-o", "--output", help="Output file (default: {kernel}.{format})."
    )
    dump_parser.add_argument(
        "-d", "--debug", action="store_true", help="Include debug information."
    )

    args = parser.parse_args()

    m, n, k = 4096, 4096, 4096

    # Define a simple kernel function for matrix multiplication
    def launch_torch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        torch.matmul(a, b, out=c)

    if args.mode == "profile":
        if args.kernel == "torch":
            profile(m, n, k, "torch", launch_torch)
        elif args.kernel == "simple":
            profile(m, n, k, "simple", launch_simple)
        elif args.kernel == "tile":
            profile(m, n, k, "tile", launch_tile)
    elif args.mode == "dump":
        if args.kernel == "simple":
            if args.format == "ptx":
                output_file = args.output if args.output else "simple.ptx"
                dump_ptx("simple", simple_src, output_file, args.debug)
            elif args.format == "sass":
                output_file = args.output if args.output else "simple.sass"
                dump_sass("simple", simple_src, output_file, args.debug)
    elif args.mode == "verify":
        # Verify the kernel
        verify(m, n, k, "torch", launch_torch)
        verify(m, n, k, "simple", launch_simple)
        verify(m, n, k, "tile", launch_tile)
    elif args.mode == "benchmark":
        # Benchmark the kernel
        benchmark(m, n, k, "torch", launch_torch)
        benchmark(m, n, k, "simple", launch_simple)
        benchmark(m, n, k, "simple", launch_tile)


if __name__ == "__main__":
    main()
