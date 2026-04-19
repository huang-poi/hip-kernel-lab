#include <hip/hip_runtime.h>
#include <iostream>

// Level 1: Naive dot product with atomic add
__global__ void dot_naive(const float* a, const float* b, float* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, a[idx] * b[idx]);
    }
}
