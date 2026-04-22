#include <hip/hip_runtime.h>

// Naive 2D convolution — direct nested loop
__global__ void conv2d_naive(const float* input, const float* kernel, float* output,
                              int H, int W, int KH, int KW, int C_in, int C_out) {
    int oc = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (oh < H && ow < W && oc < C_out) {
        float sum = 0.0f;
        for (int ic = 0; ic < C_in; ic++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    int ih = oh - KH/2 + kh;
                    int iw = ow - KW/2 + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        sum += input[ic * H * W + ih * W + iw] * kernel[oc * C_in * KH * KW + ic * KH * KW + kh * KW + kw];
                    }
                }
            }
        }
        output[oc * H * W + oh * W + ow] = sum;
    }
}
