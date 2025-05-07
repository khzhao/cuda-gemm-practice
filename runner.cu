#include "1-naive.cuh"
#include "2-global-memory-coalescing.cuh"
#include "3-shared-memory-blocking.cuh"
#include "4-1D-blocktiling.cuh"
#include "5-2D-blocktiling.cuh"
#include "6-vectorize.cuh"

#include <iostream>

#define CEIL_DIV(M, N) ((M + N - 1) / N)

void run_kernel(int kernel_id) {
  // Allocate memory the size of which we are concerned with
  const int PROBLEM_SIZE = 4096;
  const int PROBLEM_MATRIX_SIZE = PROBLEM_SIZE * PROBLEM_SIZE;
  const int PROBLEM_MATRIX_SIZE_BYTES = PROBLEM_MATRIX_SIZE * sizeof(float);

  float* h_M = new float[PROBLEM_MATRIX_SIZE_BYTES];
  float* h_C = new float[PROBLEM_MATRIX_SIZE_BYTES];

  // Initialize h_M with some values;
  for (int i = 0; i < PROBLEM_SIZE; ++i) {
    for (int j = 0; j < PROBLEM_SIZE; ++j) {
      h_M[i * PROBLEM_SIZE + j] = static_cast<float>((i + j) % 5);
    }
  }

  // Malloc on CUDA some arrays then copy over the values
  float* d_A;
  float* d_B;
  float* d_C;
  cudaMalloc(&d_A, PROBLEM_MATRIX_SIZE_BYTES);
  cudaMalloc(&d_B, PROBLEM_MATRIX_SIZE_BYTES);
  cudaMalloc(&d_C, PROBLEM_MATRIX_SIZE_BYTES);

  cudaMemcpy(d_A, h_M, PROBLEM_MATRIX_SIZE_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_M, PROBLEM_MATRIX_SIZE_BYTES, cudaMemcpyHostToDevice);

  // Now we can run our kernels
  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);

  switch (kernel_id) {
    case 1: {
      dim3 blockDim(32, 32);
      dim3 gridDim(CEIL_DIV(PROBLEM_SIZE, 32), CEIL_DIV(PROBLEM_SIZE, 32));
      gemm_naive<<<gridDim, blockDim>>>(PROBLEM_SIZE, PROBLEM_SIZE, PROBLEM_SIZE, d_A, d_B, d_C);
      break;
    }
    case 2: {
      dim3 blockDim(32, 32);
      dim3 gridDim(CEIL_DIV(PROBLEM_SIZE, 32), CEIL_DIV(PROBLEM_SIZE, 32));
      gemm_global_memory_coalescing<<<gridDim, blockDim>>>(PROBLEM_SIZE, PROBLEM_SIZE, PROBLEM_SIZE, d_A, d_B, d_C);
      break;
    }
    case 3: {
      dim3 blockDim(32, 32);
      dim3 gridDim(CEIL_DIV(PROBLEM_SIZE, 32), CEIL_DIV(PROBLEM_SIZE, 32));
      gemm_shared_memory_blocking<32><<<gridDim, blockDim>>>(PROBLEM_SIZE, PROBLEM_SIZE, PROBLEM_SIZE, d_A, d_B, d_C);
      break;
    }
    case 4: {
      const int BM = 64;
      const int BK = 8;
      const int BN = 64;
      const int TM = 8;
      dim3 gridDim(CEIL_DIV(PROBLEM_SIZE, BN), CEIL_DIV(PROBLEM_SIZE, BM));
      dim3 blockDim((BM * BN) / TM);
      gemm_1D_blocktiling<BM, BK, BN, TM><<<gridDim, blockDim>>>(PROBLEM_SIZE, PROBLEM_SIZE, PROBLEM_SIZE, d_A, d_B, d_C);
      break;
    }
    case 5: {
      const int BM = 128;
      const int BN = 128;
      const int BK = 32;
      const int TM = 8;
      const int TN = 8;
      dim3 gridDim(CEIL_DIV(PROBLEM_SIZE, BN), CEIL_DIV(PROBLEM_SIZE, BM));
      dim3 blockDim((BM * BN) / (TM * TN));
      gemm_2D_blocktiling<BM, BK, BN, TM, TN>
        <<<gridDim, blockDim>>>(PROBLEM_SIZE, PROBLEM_SIZE, PROBLEM_SIZE, d_A, d_B, d_C);
      break;
    }
    case 6: {
      const int BM = 128;
      const int BN = 128;
      const int BK = 32;
      const int TM = 8;
      const int TN = 8;
      dim3 gridDim(CEIL_DIV(PROBLEM_SIZE, BN), CEIL_DIV(PROBLEM_SIZE, BM));
      dim3 blockDim((BM * BN) / (TM * TN));
      gemm_vectorize<BM, BK, BN, TM, TN>
        <<<gridDim, blockDim>>>(PROBLEM_SIZE, PROBLEM_SIZE, PROBLEM_SIZE, d_A, d_B, d_C);
      break;
    }
    default:
      throw std::runtime_error("Unexpected kernel_id");
  }
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);

  // Now copy the result back to the host
  cudaMemcpy(h_C, d_C, PROBLEM_MATRIX_SIZE_BYTES, cudaMemcpyDeviceToHost);

  // Print out the first few entries of the matrix
  std::cout << ">>>Ran kernel_id=" << kernel_id << std::endl;
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return;
  }

  std::cout << "h_C[0]=" << h_C[0] << std::endl;
  std::cout << "h_C[1]=" << h_C[1] << std::endl;
  std::cout << "h_C[2]=" << h_C[2] << std::endl;
  std::cout << "h_C[last]=" << h_C[PROBLEM_MATRIX_SIZE - 1] << std::endl;

  float elapsed = 0.f;
  cudaEventElapsedTime(&elapsed, start_event, stop_event);
  std::cout << "Kernel ran for " << elapsed << " ms" << std::endl;

  // Free all the memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_M;
  delete[] h_C;
}

int main() {
  run_kernel(1);
  run_kernel(2);
  run_kernel(3);
  run_kernel(4);
  run_kernel(5);
  run_kernel(6);
} 