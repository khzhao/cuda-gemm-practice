#pragma once

#include <cuda_runtime.h>

// In this kernel we take advantage of the shared memory that's built into the chip, 
// located on the streaming multiprocessor. There's only one shared memory per SM
// This shared memory is partitioned across the blocks. A thread can communicate with
// other threads in the same block via the shared memory chunk.
template <int BLOCKSIZE>
__global__ void gemm_shared_memory_blocking(int M, int K, int N, float* A, float* B, float* C) {
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  auto cCol = blockIdx.x * BLOCKSIZE;
  auto cRow = blockIdx.y * BLOCKSIZE;

  // Advance the pointers to the relevant regions, we are trying to compute the values of
  // size BLOCKSIZE * BLOCKSIZE inside C
  A += cRow * K; 
  B += cCol;
  C += cRow * N + cCol;

  // Now we must compute the values of the block
  auto threadCol = threadIdx.x;
  auto threadRow = threadIdx.y;

  // We assume that BLOCKSIZE divides M, K, N and all are square matrices
  float tmp = 0.f;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] = tmp;
}