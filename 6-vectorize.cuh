#pragma once

#include <cuda_runtime.h>

#include <cassert>

#ifdef VECTORIZE_WIDTH
#undef VECTORIZE_WIDTH
#endif

#define VECTORIZE_WIDTH 2

namespace {
  using floatType = float2;
}

template <int BM, int BK, int BN, int TM, int TN>
__global__ void gemm_vectorize(int M, int K, int N, float* A, float* B, float* C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const auto cCol = blockIdx.x * BN;
  const auto cRow = blockIdx.y * BM;

  // Advance the pointers so that we are on the correct batch
  A += cRow * K;
  B += cCol;
  C += cRow * N + cCol;

  // Ok now we need to do the following, we are assigning (threadRow, threadCol)
  // per 2D batch in the outputs
  const auto MDIV = BM / TM;
  const auto NDIV = BN / TN; 
  const auto BATCH_N_PARTS = BK / NDIV;
  const auto BATCH_M_PARTS = BK / MDIV;
  assert(MDIV * NDIV == blockDim.x);
  assert(BK % NDIV == 0);
  assert(BATCH_N_PARTS % VECTORIZE_WIDTH == 0);
  assert(TN % VECTORIZE_WIDTH == 0);

  // There should be MDIV * NDIV total number of threads
  const auto threadCol = threadIdx.x % NDIV;
  const auto threadRow = threadIdx.x / NDIV;

  // Larger thread-local cache
  float cachedValues[TM * TN] = {0.f};

  // Now we need to do the actual computations
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Populate the shared memory caches, can also look at the tranpose of As to get a
    // speedup via global memory coalescing
    // Improvements: store As as the transposed version of A, this way we can iterate through
    // the rows much faster
    for (int i = 0; i < TM; ++i) {
      for (int j = 0; j < BATCH_N_PARTS / VECTORIZE_WIDTH; ++j) {
        floatType tmp = reinterpret_cast<floatType*>(&A[(threadRow * TM + i) * K + threadCol * BATCH_N_PARTS + VECTORIZE_WIDTH * j])[0];
        As[(threadRow * TM + i) * BK + threadCol * BATCH_N_PARTS + VECTORIZE_WIDTH * j] = tmp.x;
        As[(threadRow * TM + i) * BK + threadCol * BATCH_N_PARTS + VECTORIZE_WIDTH * j + 1] = tmp.y;
      }
    }
    for (int i = 0; i < BATCH_M_PARTS; ++i) {
      for (int j = 0; j < TN / VECTORIZE_WIDTH; ++j) {
        float4 tmp = reinterpret_cast<float4*>(&B[(threadRow * BATCH_M_PARTS + i) * N + threadCol * TN + VECTORIZE_WIDTH * j])[0];
        Bs[(threadRow * BATCH_M_PARTS + i) * BN + threadCol * TN + VECTORIZE_WIDTH * j] = tmp.x;
        Bs[(threadRow * BATCH_M_PARTS + i) * BN + threadCol * TN + VECTORIZE_WIDTH * j + 1] = tmp.y;
      }
    }
    __syncthreads();

    // Advance the pointers to the next batch
    A += BK;
    B += BK * N;

    // Compute the cached values
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (int i = 0; i < TM; ++i) {
        float tmpA = As[(threadRow * TM + i) * BK + dotIdx];
        for (int j = 0; j < TN; ++j) {
          float tmpB = Bs[threadCol * TN + j + dotIdx * BN];
          cachedValues[i * TN + j] += tmpA * tmpB;
        }
      }
    }
    __syncthreads();
  }

  // Now copy the actual results to the output
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN / VECTORIZE_WIDTH; ++j) {
      floatType tmp = reinterpret_cast<floatType*>(&C[(threadRow * TM + i) * N + threadCol * TN + VECTORIZE_WIDTH * j])[0];
      tmp.x = cachedValues[i * TN + j * VECTORIZE_WIDTH];
      tmp.y = cachedValues[i * TN + j * VECTORIZE_WIDTH + 1];
      reinterpret_cast<floatType*>(&C[(threadRow * TM + i) * N + threadCol * TN + VECTORIZE_WIDTH * j])[0] = tmp;
    }
  }
}
