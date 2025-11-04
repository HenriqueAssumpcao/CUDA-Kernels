#pragma once

#include <stdlib.h>

#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <driver_types.h>

/**
 * @brief SGEMM kernel with 2D thread blocks and 2D shared memory
 * @tparam BSZ_M Block size in M dimension (rows processed per thread block)
 * @tparam BSZ_K Block size in K dimension (shared dimension tile size)
 * @tparam BSZ_N Block size in N dimension (columns processed per thread block)
 * @tparam TSZ_M Thread tile size in M dimension (rows per thread)
 * @tparam TSZ_N Thread tile size in N dimension (columns per thread)
 * @param A Input matrix A (M x K) in row-major order, device memory
 * @param B Input matrix B (K x N) in row-major order, device memory
 * @param C Output matrix C (M x N) in row-major order, device memory
 * @param M Number of rows in matrices A and C
 * @param K Number of columns in A and rows in B (shared dimension)
 * @param N Number of columns in matrices B and C
 * @param alpha Scalar multiplier for A*B product
 * @param beta Scalar multiplier for existing C values
 *
 * Each thread block computes a BSZ_M x BSZ_N tile of matrix C, while each
 * thread within the block computes a TSZ_M x TSZ_N sub-tile.
 *
 * Launch configuration requirements:
 * - blockDim: (BSZ_N / TSZ_N, BSZ_M / TSZ_M, 1).
 * - gridDim: (CEIL_DIV(N, BSZ_N), CEIL_DIV(M, BSZ_M), 1)
 *
 * Template parameter constraints:
 * - BSZ_M must be divisible by TSZ_M, BSZ_N must be divisible by TSZ_N
 *
 * Performs the operation: C = alpha * A * B + beta * C
 */
template <const int BSZ_M, const int BSZ_K, const int BSZ_N, const int TSZ_M,
          const int TSZ_N>
__global__ void sgemm_threadtiling(const float *A, const float *B, float *C,
                                   int M, int K, int N, float alpha,
                                   float beta) {
    const int thread_col = threadIdx.x;
    const int thread_row = threadIdx.y;

    const int block_start_row = blockIdx.y * BSZ_M;
    const int block_start_col = blockIdx.x * BSZ_N;

    const int global_row_start = block_start_row + thread_row * TSZ_M;
    const int global_col_start = block_start_col + thread_col * TSZ_N;

    const int thread_id = thread_row * BSZ_N + thread_col;
    const int nthreads_block = BSZ_M * BSZ_N;

    __shared__ float As[BSZ_M][BSZ_K];
    __shared__ float Bs[BSZ_K][BSZ_N];

    float dot[TSZ_M][TSZ_N] = {{0.0f}};

    for (int block_tile_start = 0; block_tile_start < K;
         block_tile_start += BSZ_K) {

        // strided loads
        for (int i = thread_id; i < BSZ_M * BSZ_K; i += nthreads_block) {
            int r = i / BSZ_K;
            int c = i % BSZ_K;
            int g_row = block_start_row + r;
            int g_col = block_tile_start + c;

            As[r][c] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        for (int i = thread_id; i < BSZ_K * BSZ_N; i += nthreads_block) {
            int r = i / BSZ_N;
            int c = i % BSZ_N;
            int g_row = block_tile_start + r;
            int g_col = block_start_col + c;

            Bs[r][c] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        // outer product
        for (int k = 0; k < BSZ_K; ++k) {
            float colA[TSZ_M];
            for (int i = 0; i < TSZ_M; ++i) {
                colA[i] = As[thread_row * TSZ_M + i][k];
            }
            float rowB[TSZ_N];
            for (int j = 0; j < TSZ_N; ++j) {
                rowB[j] = Bs[k][thread_col * TSZ_N + j];
            }
            for (int i = 0; i < TSZ_M; ++i) {
                for (int j = 0; j < TSZ_N; ++j) {
                    dot[i][j] += colA[i] * rowB[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TSZ_M; ++i) {
        for (int j = 0; j < TSZ_N; ++j) {
            int global_row = global_row_start + i;
            int global_col = global_col_start + j;

            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] =
                    alpha * dot[i][j] + beta * C[global_row * N + global_col];
            }
        }
    }
}

/**
 * @brief HGEMM kernel with 2D thread blocks and 2D shared memory (FP16
 * inputs/output, FP32 accumulation)
 * @tparam BSZ_M Block size in M dimension (rows processed per thread block)
 * @tparam BSZ_K Block size in K dimension (shared dimension tile size)
 * @tparam BSZ_N Block size in N dimension (columns processed per thread block)
 * @tparam TSZ_M Thread tile size in M dimension (rows per thread)
 * @tparam TSZ_N Thread tile size in N dimension (columns per thread)
 * @param A Input matrix A (M x K) in row-major order, device memory (FP16)
 * @param B Input matrix B (K x N) in row-major order, device memory (FP16)
 * @param C Output matrix C (M x N) in row-major order, device memory (FP16)
 * @param M Number of rows in matrices A and C
 * @param K Number of columns in A and rows in B (shared dimension)
 * @param N Number of columns in matrices B and C
 * @param alpha Scalar multiplier for A*B product (FP32)
 * @param beta Scalar multiplier for existing C values (FP32)
 *
 * Each thread block computes a BSZ_M x BSZ_N tile of matrix C, while each
 * thread within the block computes a TSZ_M x TSZ_N sub-tile.
 *
 * Launch configuration requirements:
 * - blockDim: (BSZ_N / TSZ_N, BSZ_M / TSZ_M, 1).
 * - gridDim: (CEIL_DIV(N, BSZ_N), CEIL_DIV(M, BSZ_M), 1)
 *
 * Template parameter constraints:
 * - BSZ_M must be divisible by TSZ_M, BSZ_N must be divisible by TSZ_N
 *
 * Performs the operation: C = alpha * A * B + beta * C
 */
template <const int BSZ_M, const int BSZ_K, const int BSZ_N, const int TSZ_M,
          const int TSZ_N>
__global__ void hgemm_threadtiling(const half *A, const half *B, half *C, int M,
                                   int K, int N, float alpha, float beta) {
    const int thread_col = threadIdx.x;
    const int thread_row = threadIdx.y;

    const int block_start_row = blockIdx.y * BSZ_M;
    const int block_start_col = blockIdx.x * BSZ_N;

    const int global_row_start = block_start_row + thread_row * TSZ_M;
    const int global_col_start = block_start_col + thread_col * TSZ_N;

    const int thread_id = thread_row * blockDim.x + thread_col;
    const int nthreads_block = blockDim.x * blockDim.y;

    __shared__ half As[BSZ_M][BSZ_K];
    __shared__ half Bs[BSZ_K][BSZ_N];

    float dot[TSZ_M][TSZ_N] = {{0.0f}};

    for (int block_tile_start = 0; block_tile_start < K;
         block_tile_start += BSZ_K) {

        // strided loads
        for (int i = thread_id; i < BSZ_M * BSZ_K; i += nthreads_block) {
            int r = i / BSZ_K;
            int c = i % BSZ_K;
            int g_row = block_start_row + r;
            int g_col = block_tile_start + c;
            As[r][c] =
                (g_row < M && g_col < K) ? A[g_row * K + g_col] : (half)0.0f;
        }
        for (int i = thread_id; i < BSZ_K * BSZ_N; i += nthreads_block) {
            int r = i / BSZ_N;
            int c = i % BSZ_N;
            int g_row = block_tile_start + r;
            int g_col = block_start_col + c;
            Bs[r][c] =
                (g_row < K && g_col < N) ? B[g_row * N + g_col] : (half)0.0f;
        }
        __syncthreads();

        for (int k = 0; k < BSZ_K; ++k) {
            float colA[TSZ_M];
            for (int i = 0; i < TSZ_M; ++i) {
                colA[i] = __half2float(As[thread_row * TSZ_M + i][k]);
            }

            float rowB[TSZ_N];
            for (int j = 0; j < TSZ_N; ++j) {
                rowB[j] = __half2float(Bs[k][thread_col * TSZ_N + j]);
            }

            for (int i = 0; i < TSZ_M; ++i) {
                for (int j = 0; j < TSZ_N; ++j) {
                    dot[i][j] += colA[i] * rowB[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TSZ_M; ++i) {
        for (int j = 0; j < TSZ_N; ++j) {
            int global_row = global_row_start + i;
            int global_col = global_col_start + j;

            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = __float2half(
                    alpha * dot[i][j] +
                    beta * __half2float(C[global_row * N + global_col]));
            }
        }
    }
}