#include <hip/hip_runtime.h>
#include <iostream>

// Level 2: Optimized with float4 vectorized loads
__global__ void vec_add_vec4(const float4* a, const float4* b, float4* c, size_t n4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        c[idx] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
    }
}

// 4x fewer memory transactions, better memory coalescing
// Expected: ~5.3 TB/s on MI300X (vs ~2.5 TB/s for scalar)
