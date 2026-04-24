#include <hip/hip_runtime.h>

// Level 1: Atomic reduction — simple but slow for large arrays
__global__ void reduce_atomic(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(output, input[idx]);
    }
}
