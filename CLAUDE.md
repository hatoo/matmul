# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CUDA matrix multiplication benchmarking project that implements and compares different matrix multiplication kernels. The project uses PyTorch's inline CUDA extension system to compile and run custom CUDA kernels.

## Architecture

- **main.py**: Main benchmarking framework with utilities for benchmarking, verification, profiling, and code dumping
- **kernels/simple.cu**: Simple CUDA matrix multiplication kernel implementation
- **src/**: Currently empty directory for additional source files
- **Generated files**: PTX and SASS assembly dumps are generated in the root directory

## Key Components

- **Kernel Loading**: Uses `torch.utils.cpp_extension.load_inline` to dynamically compile CUDA kernels
- **Benchmarking System**: Comprehensive timing with warmup runs, multiple iterations, and statistical reporting
- **Verification**: Compares custom kernel results against PyTorch's reference implementation using `torch.allclose`
- **Code Dumping**: Utilities to extract PTX and SASS assembly from CUDA kernels using nvcc and cuobjdump

## Development Commands

### Running Benchmarks
```bash
uv run main.py benchmark          # Benchmark all kernels
uv run main.py verify            # Verify kernel correctness
uv run main.py profile torch     # Profile PyTorch kernel
uv run main.py profile simple    # Profile custom simple kernel
```

### Code Analysis
```bash
uv run main.py dump simple -f ptx    # Dump PTX assembly
uv run main.py dump simple -f sass   # Dump SASS assembly
uv run main.py dump simple -o custom.ptx  # Custom output file
uv run main.py dump simple -f ptx -d # Dump PTX with debug info
uv run main.py dump simple -f sass -d # Dump SASS with debug info
```

### Dependencies
- Install with: `uv sync`
- Requires CUDA toolkit (nvcc, cuobjdump) for assembly dumping
- Uses PyTorch nightly build from CUDA 12.8 index
- Project managed with uv (see pyproject.toml for configuration)

## CUDA Kernel Development

- Kernels are written in CUDA C++ in the `kernels/` directory
- The main.py framework provides utilities for loading, benchmarking, and analyzing kernels
- Default matrix size for benchmarking: 4096x4096x4096
- Block size for CUDA kernels: 16x16 threads