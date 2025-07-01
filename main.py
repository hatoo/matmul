import torch
from torch.utils.cpp_extension import load_inline
import time
import argparse
import os
import subprocess
from typing import Any
from abc import ABC, abstractmethod

from collections.abc import Callable


class KernelBase(ABC):
    """Base class for CUDA matrix multiplication kernels."""

    def __init__(self, name: str):
        self.name = name
        self._kernel_module = None

    @abstractmethod
    def _load_kernel(self) -> Any:
        """Load the CUDA kernel module."""
        pass

    @property
    def kernel_module(self) -> Any:
        """Get the loaded kernel module, loading it if necessary."""
        if self._kernel_module is None:
            self._kernel_module = self._load_kernel()
        return self._kernel_module

    @abstractmethod
    def launch(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        """Launch the kernel with given tensors."""
        pass

    def verify(self, m: int, n: int, k: int):
        """Verify kernel correctness against PyTorch reference."""
        a = torch.randn(m, k, dtype=torch.float32, device="cuda")
        b = torch.randn(k, n, dtype=torch.float32, device="cuda")
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
        c_ref = torch.matmul(a, b)

        self.launch(a, b, c)
        torch.cuda.synchronize()

        if torch.allclose(c, c_ref, rtol=1e-02, atol=1e-03):
            print(f"{self.name} kernel verification passed.")
            return True
        else:
            print(f"{self.name} kernel verification failed.")
            return False

    def benchmark(self, n: int, m: int, k: int, warmup_times: int = 4, times: int = 8):
        """Benchmark the kernel performance."""
        a = torch.randn(n, k, dtype=torch.float32, device="cuda")
        b = torch.randn(k, m, dtype=torch.float32, device="cuda")
        c = torch.zeros(n, m, dtype=torch.float32, device="cuda")

        # Warmup
        for _ in range(warmup_times):
            torch.cuda.synchronize()
            self.launch(a, b, c)
            torch.cuda.synchronize()

        # Timed runs
        elapsed_times = []
        for _ in range(times):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            self.launch(a, b, c)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            elapsed_times.append(end - start)

        average_time = sum(elapsed_times) / times
        min_time = min(elapsed_times)
        max_time = max(elapsed_times)

        print(
            f"{self.name} kernel: {n}x{m}x{k} avg: {average_time / 1e6:.2f} ms, "
            f"min: {min_time / 1e6:.2f} ms, max: {max_time / 1e6:.2f} ms."
        )

        return average_time, min_time, max_time

    def profile(self, m: int, n: int, k: int):
        """Profile the kernel with a single run."""
        a = torch.randn(m, k, dtype=torch.float32, device="cuda")
        b = torch.randn(k, n, dtype=torch.float32, device="cuda")
        c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        self.launch(a, b, c)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()

        elapsed = (end - start) / 1e6
        print(
            f"{self.name} kernel: {n}x{m}x{k} took {elapsed:.2f} ms (profile mode, single run)."
        )
        return elapsed

    def dump_ptx(self, output_file: str | None = None, debug: bool = False):
        """Dump PTX code for this kernel."""
        if not hasattr(self, "cuda_source"):
            raise NotImplementedError(
                f"PTX dumping not supported for {self.name} kernel"
            )

        if output_file is None:
            output_file = f"{self.name}.ptx"
        dump_ptx(self.name, getattr(self, "cuda_source"), output_file, debug)

    def dump_sass(self, output_file: str | None = None, debug: bool = False):
        """Dump SASS code for this kernel."""
        if not hasattr(self, "cuda_source"):
            raise NotImplementedError(
                f"SASS dumping not supported for {self.name} kernel"
            )

        if output_file is None:
            output_file = f"{self.name}.sass"
        dump_sass(self.name, getattr(self, "cuda_source"), output_file, debug)


class TorchKernel(KernelBase):
    """PyTorch reference implementation."""

    def __init__(self):
        super().__init__("torch")

    def _load_kernel(self) -> Any:
        return None  # No CUDA module needed for PyTorch

    def launch(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        torch.matmul(a, b, out=c)


class SimpleKernel(KernelBase):
    """Simple CUDA kernel implementation."""

    def __init__(self):
        super().__init__("simple")
        with open("kernels/simple.cu", "r") as f:
            self.cuda_source = f.read()

    def _load_kernel(self) -> Any:
        return load_inline(
            name="simple",
            cpp_sources="void simple(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k);",
            cuda_sources=self.cuda_source,
            functions="simple",
            with_cuda=True,
            extra_cuda_cflags=["-O3"],
        )

    def launch(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        n, k = a.shape
        _k, m = b.shape
        self.kernel_module.simple(a.data_ptr(), b.data_ptr(), c.data_ptr(), n, m, k)


class TileKernel(KernelBase):
    """Tiled CUDA kernel implementation."""

    def __init__(self):
        super().__init__("tile")
        with open("kernels/tile.cu", "r") as f:
            self.cuda_source = f.read()

    def _load_kernel(self) -> Any:
        return load_inline(
            name="tile",
            cpp_sources="void tile(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k);",
            cuda_sources=self.cuda_source,
            functions="tile",
            with_cuda=True,
            extra_cuda_cflags=["-O3"],
        )

    def launch(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        n, k = a.shape
        _k, m = b.shape
        self.kernel_module.tile(a.data_ptr(), b.data_ptr(), c.data_ptr(), n, m, k)


class Cute01Kernel(KernelBase):
    """Cute 01 CUDA kernel implementation."""

    def __init__(self):
        super().__init__("cute01")
        with open("kernels/cute01.cu", "r") as f:
            self.cuda_source = f.read()

    def _load_kernel(self) -> Any:
        cutlass_path = os.environ["CUTLASS_PATH"]

        return load_inline(
            name="cute01",
            cpp_sources="void cute01(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k);",
            cuda_sources=self.cuda_source,
            functions="cute01",
            with_cuda=True,
            extra_cuda_cflags=["-O3"],
            extra_include_paths=[cutlass_path + "/include"],
        )

    def launch(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        m, k = a.shape
        _k, n = b.shape
        self.kernel_module.cute01(a.data_ptr(), b.data_ptr(), c.data_ptr(), m, n, k)


class Cute02Kernel(KernelBase):
    """Cute 02 CUDA kernel implementation."""

    def __init__(self):
        super().__init__("cute02")
        with open("kernels/cute02.cu", "r") as f:
            self.cuda_source = f.read()

    def _load_kernel(self) -> Any:
        cutlass_path = os.environ["CUTLASS_PATH"]

        return load_inline(
            name="cute02",
            cpp_sources="void cute02(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k);",
            cuda_sources=self.cuda_source,
            functions="cute02",
            with_cuda=True,
            extra_cuda_cflags=["-O3"],
            extra_include_paths=[cutlass_path + "/include"],
        )

    def launch(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        m, k = a.shape
        _k, n = b.shape
        self.kernel_module.cute02(a.data_ptr(), b.data_ptr(), c.data_ptr(), m, n, k)


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
        cutlass_path = os.environ["CUTLASS_PATH"]
        # Use nvcc to compile to PTX
        cmd = [
            "nvcc",
            "--ptx",
            "-O3" if not debug else "-O0",
            "-arch=sm_120",  # adjust based on your GPU architecture
            "--use_fast_math",
            "-o",
            output_file,
            f"-I{cutlass_path}/include",
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


def get_available_kernels():
    """Get a dictionary of all available kernel implementations."""
    return {
        "torch": TorchKernel(),
        "simple": SimpleKernel(),
        "tile": TileKernel(),
        "cute01": Cute01Kernel(),
        "cute02": Cute02Kernel(),
    }


# Define kernel choices to avoid repetition
def get_kernel_choices():
    """Get kernel choices from available kernels."""
    kernels = get_available_kernels()
    all_choices = list(kernels.keys())
    # Dumpable kernels are those that have CUDA source (exclude torch)
    dumpable_choices = [name for name in all_choices if name != "torch"]
    return all_choices, dumpable_choices


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA kernels.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Get kernel choices
    all_kernel_choices, dumpable_kernel_choices = get_kernel_choices()

    # Benchmark mode
    subparsers.add_parser("benchmark", help="Benchmark all kernels.")

    # Verify mode
    verify_parser = subparsers.add_parser("verify", help="Verify kernels.")
    verify_parser.add_argument(
        "kernel",
        nargs="?",
        choices=all_kernel_choices,
        help="Kernel to verify (optional, verifies all if not specified).",
    )

    # Profile mode
    profile_parser = subparsers.add_parser(
        "profile", help="Profile a selected kernel (single run)."
    )
    profile_parser.add_argument(
        "kernel",
        choices=all_kernel_choices,
        help="Kernel to profile.",
    )

    # Dump mode (PTX or SASS)
    dump_parser = subparsers.add_parser(
        "dump", help="Dump PTX or SASS code for a selected kernel."
    )
    dump_parser.add_argument(
        "kernel", choices=dumpable_kernel_choices, help="Kernel to dump code for."
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

    # Initialize kernel instances
    kernels = get_available_kernels()

    if args.mode == "profile":
        kernel = kernels[args.kernel]
        kernel.profile(m, n, k)
    elif args.mode == "dump":
        kernel = kernels[args.kernel]
        if args.format == "ptx":
            output_file = args.output if args.output else f"{args.kernel}.ptx"
            kernel.dump_ptx(output_file, args.debug)
        elif args.format == "sass":
            output_file = args.output if args.output else f"{args.kernel}.sass"
            kernel.dump_sass(output_file, args.debug)
    elif args.mode == "verify":
        # Verify specified kernel or all kernels
        if args.kernel:
            kernel = kernels[args.kernel]
            kernel.verify(m, n, k)
        else:
            for name, kernel in kernels.items():
                kernel.verify(m, n, k)
    elif args.mode == "benchmark":
        # Benchmark all kernels
        for name, kernel in kernels.items():
            kernel.benchmark(n, m, k)


if __name__ == "__main__":
    main()
