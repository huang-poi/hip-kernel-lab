#include <hip/hip_runtime.h>

// Level 3: Warp shuffle reduction (no shared memory needed)
__device__ float warp_reduce_sum(float val) {
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void dot_warp_shuffle(const float* a, const float* b, float* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? a[idx] * b[idx] : 0.0f;
    
    val = warp_reduce_sum(val);
    
    // First thread in each warp accumulates to shared
    __shared__ float warp_sums[32]; // max 32 warps per block
    int lane = threadIdx.x % 64; // CDNA wavefront = 64
    int warp_id = threadIdx.x / 64;
    
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        val = (lane < blockDim.x / 64) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) atomicAdd(result, val);
    }
}
