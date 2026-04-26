#include <hip/hip_runtime.h>

// Privatized histogram — each block has its own histogram in shared memory
// Final merge via atomicAdd to global histogram
// Much faster than global atomics for non-uniform distributions

__global__ void histogram_privatized(const uint8_t* input, uint32_t* output, int n, int bins) {
    extern __shared__ uint32_t local_hist[];
    
    int tid = threadIdx.x;
    for (int i = tid; i < bins; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) {
        atomicAdd(&local_hist[input[idx]], 1);
    }
    __syncthreads();
    
    for (int i = tid; i < bins; i += blockDim.x) {
        atomicAdd(&output[i], local_hist[i]);
    }
}
