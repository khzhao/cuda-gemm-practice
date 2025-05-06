#include <cuda_runtime.h>

// On a high level, we want consecutive threadIds to access consecutive memory addresses
// Order does matter here so even inside a warp, if there is a permutation where the accesses
// are consecutive, it doesn't matter. It needs to be perfectly consecutive. The trouble is
// mapping the thread indices to units of work in the matrix
// Keep in mind that we want nearby indices to belong to the same block as best as possible
// This is so that they can access the same shared memory
__global__ void gemm_global_memory_coalescing(int M, int K, int N, float* A, float* B, float* C) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.f;
    for (int i = 0; i < K; ++i) {
      tmp += A[y * K + i] * B[i * N + x];
    }
    C[y * N + x] = tmp;
  }
}
