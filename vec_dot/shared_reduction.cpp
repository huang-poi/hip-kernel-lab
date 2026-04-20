#include <hip/hip_runtime.h>
#include <iostream>

// Level 2: Shared memory reduction (much faster than atomics)
__global__ void dot_shared(const float* a, const float* b, float* result, size_t n) {
    extern __shared__ float sdata[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}
