#include <cuda_runtime.h>

// This is not good for global memory coalescing because the last index which is
// threadIdx.x is not consecutively accessing memory addresses
__global__ void gemm_naive(int M, int K, int N, float* A, float* B, float* C) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.f;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = tmp;
  }
};