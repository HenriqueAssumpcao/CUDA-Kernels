#pragma once

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <driver_types.h>

#include "sgemm.cuh"
#include "attn.cuh"
#include "utils.cuh"

/**
 * @brief CPU implementation of single-precision general matrix multiplication
 * (SGEMM)
 * @param A Input matrix A (M x K) in row-major order
 * @param B Input matrix B (K x N) in row-major order
 * @param C Output matrix C (M x N) in row-major order
 * @param M Number of rows in matrices A and C
 * @param K Number of columns in A and rows in B (shared dimension)
 * @param N Number of columns in matrices B and C
 * @param alpha Scalar multiplier for A*B product (default: 1.0)
 * @param beta Scalar multiplier for existing C values (default: 0.0)
 *
 * Performs the operation C = alpha * A * B + beta * C using a triple nested
 * loop. This serves as a reference implementation for validating GPU kernels.
 */
void run_sgemm_cpu(const float *A, const float *B, float *C, int M, int K, int N,
               float alpha, float beta) {

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double dot = 0.0;
            for (int k = 0; k < K; ++k) {
                dot += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * dot + beta * C[i * N + j];
        }
    }
}

/**
 * @brief Host wrapper for memory-coalesced naive SGEMM kernel launch
 * @tparam BSZ Block size (total number of threads per block)
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
template <const uint BSZ>
__host__ void run_sgemm_naive(const float *A, const float *B, float *C,
                                 int M, int K, int N, float alpha, float beta) {
    dim3 gridDim(CEIL_DIV(N, BSZ), CEIL_DIV(M, BSZ), 1);
    dim3 blockDim(BSZ * BSZ, 1, 1);

    sgemm_naive<BSZ>
        <<<gridDim, blockDim>>>(A, B, C, M, K, N, alpha, beta);
}

/**
 * @brief Host wrapper for block-tiled SGEMM kernel launch
 * @tparam BSZ_M Block size in M dimension (rows)
 * @tparam BSZ_K Block size in K dimension (shared dimension)
 * @tparam BSZ_N Block size in N dimension (columns)
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
template <const uint BSZ_M, const uint BSZ_K, const uint BSZ_N>
__host__ void run_sgemm_blocktiling(const float *A, const float *B, float *C,
                                    int M, int K, int N, float alpha,
                                    float beta) {
    dim3 gridDim(CEIL_DIV(N, BSZ_N), CEIL_DIV(M, BSZ_M), 1);
    dim3 blockDim(BSZ_M * BSZ_N, 1, 1);

    sgemm_blocktiling<BSZ_M, BSZ_K, BSZ_N>
        <<<gridDim, blockDim>>>(A, B, C, M, K, N, alpha, beta);
}

/**
 * @brief Host wrapper for thread-tiled SGEMM kernel launch
 * @tparam BSZ_M Block size in M dimension (rows)
 * @tparam BSZ_K Block size in K dimension (shared dimension)
 * @tparam BSZ_N Block size in N dimension (columns)
 * @tparam TSZ_M Thread tile size in M dimension
 * @tparam TSZ_N Thread tile size in N dimension
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
 * multiple output elements (TSZ_M x TSZ_N tile). Computes C = alpha * A * B + beta *
 * C.
 */
template <const uint BSZ_M, const uint BSZ_K, const uint BSZ_N, const uint TSZ_M,
          const uint TSZ_N>
__host__ void run_sgemm_blocktiling(const float *A, const float *B, float *C,
                                    int M, int K, int N, float alpha,
                                    float beta) {
    dim3 block_dim((BSZ_N / TSZ_N) * (BSZ_M / TSZ_M), 1, 1);
    dim3 grid_dim(CEIL_DIV(N, BSZ_N), CEIL_DIV(M, BSZ_M), 1);

    sgemm_threadtiling<BSZ_M, BSZ_K, BSZ_N, TSZ_M, TSZ_N>
        <<<grid_dim, block_dim>>>(A, B, C, M, K, N, alpha, beta);
}