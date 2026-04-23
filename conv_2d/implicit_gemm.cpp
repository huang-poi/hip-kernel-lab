#include <hip/hip_runtime.h>

// Implicit GEMM convolution — reuses optimized GEMM hardware
// Reshapes convolution as matrix multiply: im2col(A) * B
// Enables use of tensor cores / MFMA for convolution
