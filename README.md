# hip-kernel-lab

A collection of HIP kernel implementations with optimization notes for AMD GPUs. From basic to advanced patterns — each kernel includes multiple optimization levels with detailed performance analysis.

## Kernel Categories

### Basic Patterns
- `vec_add/` — Vector addition (introduction to HIP)
- `vec_dot/` — Dot product with shared memory reduction
- `mat_transpose/` — Matrix transpose with bank conflict avoidance

### Compute-Intensive
- `mat_mul/` — Matrix multiply (naive → tiled → shared memory → MFMA)
- `conv_2d/` — 2D convolution (naive → im2col → implicit GEMM)
- `fft/` — Fast Fourier Transform (Cooley-Tukey radix-2)

### Reduction Patterns
- `reduce_sum/` — Parallel sum (atomic → shared → warp shuffle)
- `reduce_max/` — Parallel max with index
- `histogram/` — Shared memory histogram with privatization

### Advanced
- `flash_attn/` — Flash Attention for transformers
- `sort/` — Radix sort (LSD variant)
- `scan/` — Blelloch prefix sum / scan
- `stream_compaction/` — Predicate-based compaction

## Optimization Levels

Each kernel includes 3 optimization levels:
1. **Naive**: Direct implementation, easy to understand
2. **Optimized**: Shared memory, coalescing, bank conflict avoidance
3. **Expert**: MFMA intrinsics, async copies, software pipelining

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Run & Profile

```bash
# Run all kernels
./build/run_all

# Run with profiling
rocprobe profile --app ./build/mat_mul_expert --format json
```

## Related Projects

Part of the [ROCm Developer Toolkit](https://github.com/huang-poi) ecosystem.

## License

MIT
