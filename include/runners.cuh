#pragma once

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <driver_types.h>

#include "kernel.cuh"

#define CEIL_DIV(x, y) (x + y - 1) / y

/**
 * @brief Host wrapper for memory-coalesced naive SGEMM kernel launch
 * @tparam BLOCK_SZ Block size (total number of threads per block)
 * @param A Input matrix A (M x K) in device memory
 * @param B Input matrix B (K x N) in device memory
 * @param C Output matrix C (M x N) in device memory
 * @param M Number of rows in matrices A and C
 * @param K Number of columns in A and rows in B
 * @param N Number of columns in matrices B and C
 * @param alpha Scalar multiplier for A*B product
 * @param beta Scalar multiplier for existing C values
 *
 * Launches a SGEMM implementation with improved memory coalescing
 * using 1D thread blocks. Computes C = alpha * A * B + beta * C.
 */
template <const uint BLOCK_SZ>
__host__ void run_sgemm_naive(const float *A, const float *B, float *C,
                                 int M, int K, int N, float alpha, float beta) {
    dim3 gridDim(CEIL_DIV(N, BLOCK_SZ), CEIL_DIV(M, BLOCK_SZ), 1);
    dim3 blockDim(BLOCK_SZ * BLOCK_SZ, 1, 1);

    sgemm_naive<BLOCK_SZ>
        <<<gridDim, blockDim>>>(A, B, C, M, K, N, alpha, beta);
}

/**
 * @brief Host wrapper for block-tiled SGEMM kernel launch
 * @tparam BM Block size in M dimension (rows)
 * @tparam BK Block size in K dimension (shared dimension)
 * @tparam BN Block size in N dimension (columns)
 * @param A Input matrix A (M x K) in device memory
 * @param B Input matrix B (K x N) in device memory
 * @param C Output matrix C (M x N) in device memory
 * @param M Number of rows in matrices A and C
 * @param K Number of columns in A and rows in B
 * @param N Number of columns in matrices B and C
 * @param alpha Scalar multiplier for A*B product
 * @param beta Scalar multiplier for existing C values
 *
 * Launches a block-tiled SGEMM implementation that uses shared memory
 * to improve data reuse and reduce global memory accesses.
 * Computes C = alpha * A * B + beta * C.
 */
template <const uint BM, const uint BK, const uint BN>
__host__ void run_sgemm_blocktiling(const float *A, const float *B, float *C,
                                    int M, int K, int N, float alpha,
                                    float beta) {
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);
    dim3 blockDim(BM * BN, 1, 1);

    sgemm_blocktiling<BM, BK, BN>
        <<<gridDim, blockDim>>>(A, B, C, M, K, N, alpha, beta);
}

/**
 * @brief Host wrapper for thread-tiled SGEMM kernel launch
 * @tparam BM Block size in M dimension (rows)
 * @tparam BK Block size in K dimension (shared dimension)
 * @tparam BN Block size in N dimension (columns)
 * @tparam TM Thread tile size in M dimension
 * @tparam TN Thread tile size in N dimension
 * @param A Input matrix A (M x K) in device memory
 * @param B Input matrix B (K x N) in device memory
 * @param C Output matrix C (M x N) in device memory
 * @param M Number of rows in matrices A and C
 * @param K Number of columns in A and rows in B
 * @param N Number of columns in matrices B and C
 * @param alpha Scalar multiplier for A*B product
 * @param beta Scalar multiplier for existing C values
 *
 * Launches an advanced SGEMM implementation combining block tiling and
 * thread-level tiling for maximum performance. Each thread computes
 * multiple output elements (TM x TN tile). Computes C = alpha * A * B + beta *
 * C.
 */
template <const uint BM, const uint BK, const uint BN, const uint TM,
          const uint TN>
__host__ void run_sgemm_blocktiling(const float *A, const float *B, float *C,
                                    int M, int K, int N, float alpha,
                                    float beta) {
    dim3 block_dim((BN / TN) * (BM / TM), 1, 1);
    dim3 grid_dim(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);

    sgemm_threadtiling<BM, BK, BN, TM, TN>
        <<<grid_dim, block_dim>>>(A, B, C, M, K, N, alpha, beta);
}