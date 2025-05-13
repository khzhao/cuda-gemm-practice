#pragma once

#include <cuda_runtime.h>

#include <cassert>

// Previously, we dispatched based on 
// blockTile
// - threadTile
// Now we dispatch based on warps
// blockTile
// - warpTile
//   - threadTile
// The templating is misleading because it should be BLOCK, WARP, THREAD
// By doing so, we guarantee that the threads are split correctly into their
// warps, so this means that we can take advantage of the GPU's instruction
// level parallelism (ILP)
template <int BM, int BK, int BN, int TM, int TN>
__global__ void gemm_warptiling(int M, int K, int N, float* A, float* B, float* C) {
  static constexpr int WARPSIZE = 32;

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

  // Warp level indexing math
  const int WM = TM * 16;
  const int WN = TN * 16;
  assert(BM % WM == 0);
  assert(BN % WN == 0);
  const int WMITER = BM / WM;
  const int WNITER = BN / WN;

  // Which TM * TN block this thread is responsible inside of the warp tile
  const int threadCol = threadIdx.x % (WN / TN);
  const int threadRow = threadIdx.x / (WN / TN);
  assert(WM * WN / (TM * TN) == blockDim.x);

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

  float threadResults[WMITER * WNITER][TM * TN] = {0.f};
  float regM[TM] = {0.f};
  float regN[TN] = {0.f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // Populate the SMEM caches and load A^T instead
    for (int loadOffset = 0; loadOffset < BM; loadOffset += numEntriesA) {
      float4 tmp = reinterpret_cast<float4*>(&A[(innerRowA + loadOffset) * K + innerColA * 4])[0];
      As[(innerColA * 4) * BM + innerRowA + loadOffset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + loadOffset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + loadOffset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + loadOffset] = tmp.w;
    }
    for (int loadOffset = 0; loadOffset < BK; loadOffset += numEntriesB) {
      reinterpret_cast<float4*>(&Bs[(innerRowB + loadOffset) * BN + innerColB * 4])[0] = 
        reinterpret_cast<float4*>(&B[(innerRowB + loadOffset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    // Advance the pointers
    A += BK;
    B += BK * N;

    // Compute the relevant values
    for (int wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (int wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // Load into the register values
          for (int i = 0; i < TM; i += 4) {
            float4 tmp = reinterpret_cast<float4*>(&As[dotIdx * BM + wmIdx * WM + threadRow * TM + i])[0];
            regM[i] = tmp.x;
            regM[i + 1] = tmp.y;
            regM[i + 2] = tmp.z;
            regM[i + 3] = tmp.w;
          }
          for (int i = 0; i < TN; ++i) {
            float4 tmp = reinterpret_cast<float4*>(&Bs[dotIdx * BN + wnIdx * WN + threadCol * TN + i])[0];
            regN[i] = tmp.x;
            regN[i + 1] = tmp.y;
            regN[i + 2] = tmp.z;
            regN[i + 3] = tmp.w;
          }

          for (int rowIdx = 0; rowIdx < TM; ++rowIdx) {
            for (int colIdx = 0; colIdx < TN; ++colIdx) {
              threadResults[wmIdx * WNITER + wnIdx][rowIdx * TN + colIdx] += 
                regM[rowIdx] * regN[colIdx];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // Write out the results into the corresponding dictionary
  for (int wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (int wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float* C_tmp = C + wmIdx * WM * N + wnIdx * WN;
      for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
          float4 tmp = reinterpret_cast<float4*>(&C_tmp[(threadRow * TM + i) * N + threadCol * TN + j])[0];
          tmp.x = threadResults[wmIdx * WNITER + wnIdx][i * TN + j];
          tmp.y = threadResults[wmIdx * WNITER + wnIdx][i * TN + j + 1];
          tmp.z = threadResults[wmIdx * WNITER + wnIdx][i * TN + j + 2];
          tmp.w = threadResults[wmIdx * WNITER + wnIdx][i * TN + j + 3];
          reinterpret_cast<float4*>(&C_tmp[(threadRow * TM + i) * N + threadCol * TN + j])[0] = tmp;
        }
      }
    }
  }
}
