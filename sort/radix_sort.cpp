#include <hip/hip_runtime.h>

// LSD Radix Sort for GPU
// Sorts by processing bits from LSB to MSB
// Each pass: histogram → prefix sum → scatter

// 4-bit radix (16 buckets) for good occupancy
// Expected throughput: ~10-20 billion keys/sec on MI300X

__global__ void radix_histogram(const uint32_t* input, uint32_t* histogram, int n, int bit) {
    extern __shared__ uint32_t local_hist[];
    int tid = threadIdx.x;
    local_hist[tid] = 0;
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t digit = (input[idx] >> bit) & 0xF;
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();
    
    if (tid < 16) {
        atomicAdd(&histogram[tid], local_hist[tid]);
    }
}
