#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define HIP_CHECK(call) do { hipError_t e = call; if(e!=hipSuccess){std::cerr<<hipGetErrorString(e);exit(1);} } while(0)

// Level 1: Naive vector addition
__global__ void vec_add_naive(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const size_t n = 1 << 24;
    const size_t bytes = n * sizeof(float);
    
    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_c(n);
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    vec_add_naive<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    double gbs = (3.0 * bytes / (ms * 1e-3)) / 1e9;
    
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));
    bool pass = true;
    for (size_t i = 0; i < n; i++) { if (h_c[i] != 3.0f) { pass = false; break; } }
    
    std::cout << "vec_add_naive: " << ms << " ms, " << gbs << " GB/s, " << (pass?"PASS":"FAIL") << std::endl;
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    return pass ? 0 : 1;
}
