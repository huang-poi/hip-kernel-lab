# HIP Kernel Optimization Patterns for MI300X

## Memory Hierarchy
| Level | Size | Bandwidth | Latency |
|-------|------|-----------|---------|
| Registers | 1024 VGPRs/CU | ∞ | 1 cycle |
| LDS (Shared) | 64KB/CU | ~30 TB/s | ~20 cycles |
| L2 Cache | 256MB | ~12 TB/s | ~100 cycles |
| HBM3 | 192GB | 5.3 TB/s | ~200 cycles |

## Pattern 1: Tiling
Break work into tiles that fit in LDS. Load once, reuse many times.

## Pattern 2: Vectorization
Use float4/int4 loads to reduce memory transactions by 4x.

## Pattern 3: Warp Primitives
Use __shfl_down, __shfl_xor for warp-level communication without LDS.

## Pattern 4: Software Pipelining
Overlap global memory loads with compute using double buffering.

## Pattern 5: MFMA Utilization
Use matrix core instructions for matrix operations (4-8x throughput).
