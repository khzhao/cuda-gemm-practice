#pragma once

#include <cuda_runtime.h>

#include <cassert>

// This kernel implements 1D blocktiling
// We partition the blocks into BM, BK, BN, and TM
template <int BM, int BK, int BN, int TM>
__global__ void gemm_1D_blocktiling(int M, int K, int N, float* A, float* B, float* C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const auto cCol = blockIdx.x * BN;
  const auto cRow = blockIdx.y * BM;

  // Advance the pointers so that we are on the correct batch
  A += cRow * K;
  B += cCol;
  C += cRow * N + cCol;

  // This determines which column of C to compute which is given by
  // (threadRow, threadCol)
  assert(BM * BK == blockDim.x);
  assert(BK * BN == blockDim.x);
  const auto threadCol = threadIdx.x % BN;
  const auto threadRow = threadIdx.x / BN;

  const auto innerColA = threadIdx.x % BK;
  const auto innerRowA = threadIdx.x / BK;
  const auto innerColB = threadIdx.x % BN;
  const auto innerRowB = threadIdx.x / BN;

  // Create a thread-local cache for computing an entire column
  float threadValues[TM] = {0.f};

  // Now populate the caches and iterate through the batches
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // Advance pointers for next batch
    A += BK; 
    B += BK * N;

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        float tmpA = As[(threadRow * TM + resIdx) * BK + dotIdx];
        threadValues[resIdx] += tmpA * tmpB;
      }
    }
    __syncthreads();
  }

  // Now the results to global memory
  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] = threadValues[resIdx];
  }
}
