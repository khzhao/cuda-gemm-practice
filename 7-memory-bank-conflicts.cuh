#pragma once

#include <cuda_runtime.h>

#include <cassert>

// Each bank is 4 bytes so it works out here
__device__ inline int computeSwizzleIndex(int x, int y, int N) {
  int offset = y * N + x;
  int bankIdx = offset / 32;
  int startingIdx = bankIdx * 32;
  // XOR is nice because of the various properties it exhibits
  // It's bijective and for the function with fixed y
  // f(x) = x ^ y
  // where x and y are K bit integers
  // f(x) is a permutation of [0, 2^K - 1]. 
  // XOR is it's own inverse and it's relatively uniform
  int newBank = (x ^ y) % 32;
  return startingIdx + newBank;
}

template <int BM, int BK, int BN, int TM, int TN>
__global__ void gemm_memory_bank_conflicts(int M, int K, int N, float* A, float* B, float* C) {
  // Create the shared memory for this block and populate it later
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Advance the pointers, so we are pointed to the current block of C
  // that we are trying to compute and the beginning of where A and B 
  // need to be pointing to
  const auto cCol = blockIdx.x * BN;
  const auto cRow = blockIdx.y * BM;
  A += cRow * K;
  B += cCol;
  C += cRow * N + cCol;

  // Get the current threadRow and threadCol, to see which exact subportion
  // of the batch of C we are trying to compute (TM * TN) size
  const auto MDIV = BM / TM;
  const auto NDIV = BN / TN;
  const auto threadCol = threadIdx.x % NDIV;
  const auto threadRow = threadIdx.x / NDIV;
  assert(blockDim.x == MDIV * NDIV);
  assert((BM % TM == 0) && (BN % TN == 0));

  // There are BM * BK elements in the batch inside of A that we need to load
  // with NUM_THREADS threads. Similar logic for B
  const auto NUM_THREADS = blockDim.x;
  assert(NUM_THREADS % BK == 0);
  assert(NUM_THREADS % BN == 0);
  const auto innerRowA = threadIdx.x / (BK / 4);
  const auto innerColA = threadIdx.x % (BK / 4);
  const auto innerRowB = threadIdx.x / (BN / 4);
  const auto innerColB = threadIdx.x % (BN / 4);

  // Find out how many entries each thread is responsible for
  const auto numEntriesA = NUM_THREADS / (BK / 4);
  const auto numEntriesB = NUM_THREADS / (BN / 4);

  // Create a register for storing the tile values
  float cachedValues[TM * TN] = {0.f};

  // Iterate through the blocks and try to populate the shared memory
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Populate the caches
    for (int loadOffset = 0; loadOffset < BM; loadOffset += numEntriesA) {
      float4 tmp = reinterpret_cast<float4*>(&A[(innerRowA + loadOffset) * K + innerColA * 4])[0];
      As[(innerColA * 4) * BM + innerRowA + loadOffset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + loadOffset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + loadOffset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + loadOffset] = tmp.w;
    }
    for (int loadOffset = 0; loadOffset < BK; loadOffset += numEntriesB) {
      for (int i = 0; i < 4; ++i) {
        auto x = innerColB * 4 + i;
        auto y = innerRowB + loadOffset;
        auto swz_idx = computeSwizzleIndex(x, y, BN);
        Bs[swz_idx] = B[(innerRowB + loadOffset) * N + innerColB * 4 + i];
      }
    }
    __syncthreads();

    // Advance the pointers
    A += BK;
    B += BK * N;

    // Now compute the cachedValues
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (int i = 0; i < TM / 4; ++i) {
        float4 tmpA = reinterpret_cast<float4*>(&As[dotIdx * BM + threadRow * TM + 4 * i])[0]; 
        for (int j = 0; j < TN; ++j) {
          auto x = threadCol * TN + j;
          auto y = dotIdx;
          auto tmpB = Bs[computeSwizzleIndex(x, y, BN)];
          cachedValues[(4 * i) * TN + j] += tmpA.x * tmpB;
          cachedValues[(4 * i + 1) * TN + j] += tmpA.y * tmpB;
          cachedValues[(4 * i + 2) * TN + j] += tmpA.z * tmpB;
          cachedValues[(4 * i + 3) * TN + j] += tmpA.w * tmpB;
        }
      }
    }
    __syncthreads();
  }
  
  // Now take cachedValues and populate C
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN / 4; ++j) {
      float4 tmp = reinterpret_cast<float4*>(&C[(threadRow * TM + i) * N + threadCol * TN + 4 * j])[0];
      tmp.x = cachedValues[i * TN + 4 * j];
      tmp.y = cachedValues[i * TN + 4 * j + 1];
      tmp.z = cachedValues[i * TN + 4 * j + 2];
      tmp.w = cachedValues[i * TN + 4 * j + 3];
      reinterpret_cast<float4*>(&C[(threadRow * TM + i) * N + threadCol * TN + 4 * j])[0] = tmp;
    }
  }
}
