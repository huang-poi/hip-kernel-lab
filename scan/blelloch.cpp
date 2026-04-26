#include <hip/hip_runtime.h>

// Blelloch prefix sum (exclusive scan)
// Work-efficient: O(n) work, O(log n) depth
// Two phases: up-sweep (reduce) and down-sweep)

__global__ void blelloch_scan(float* data, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int offset = 1;
    
    temp[2 * tid] = (2 * tid < n) ? data[2 * tid] : 0.0f;
    temp[2 * tid + 1] = (2 * tid + 1 < n) ? data[2 * tid + 1] : 0.0f;
    __syncthreads();
    
    // Up-sweep
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }
    
    if (tid == 0) temp[n - 1] = 0.0f;
    
    // Down-sweep
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    if (2 * tid < n) data[2 * tid] = temp[2 * tid];
    if (2 * tid + 1 < n) data[2 * tid + 1] = temp[2 * tid + 1];
}
