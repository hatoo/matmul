import argparse
from kernels import get_available_kernels

# torch.backends.cuda.matmul.allow_tf32 = True


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
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark kernels.")
    benchmark_parser.add_argument(
        "kernel",
        nargs="?",
        choices=all_kernel_choices,
        help="Kernel to benchmark (optional, benchmarks all if not specified).",
    )

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
        # Benchmark specified kernel or all kernels
        if args.kernel:
            kernel = kernels[args.kernel]
            kernel.benchmark(n, m, k)
        else:
            for name, kernel in kernels.items():
                kernel.benchmark(n, m, k)


if __name__ == "__main__":
    main()
