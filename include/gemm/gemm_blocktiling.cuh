#pragma once

#include <stdlib.h>

#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <driver_types.h>

/**
 * @brief Block-tiled SGEMM kernel using shared memory for improved performance
 * @tparam BSZ_M Block size in M dimension (rows processed per thread block)
 * @tparam BSZ_K Block size in K dimension (shared dimension tile size)
 * @tparam BSZ_N Block size in N dimension (columns processed per thread block)
 * @param A Input matrix A (M x K) in row-major order, device memory
 * @param B Input matrix B (K x N) in row-major order, device memory
 * @param C Output matrix C (M x N) in row-major order, device memory
 * @param M Number of rows in matrices A and C
 * @param K Number of columns in A and rows in B (shared dimension)
 * @param N Number of columns in matrices B and C
 * @param alpha Scalar multiplier for A*B product
 * @param beta Scalar multiplier for existing C values
 *
 * This kernel uses block tiling with shared memory to improve data reuse and
 * reduce global memory bandwidth requirements. Each thread block computes a
 * BSZ_M x BSZ_N tile of the output matrix C by processing the shared K
 * dimension in chunks of size BSZ_K.
 *
 * Launch configuration requirements:
 * - blockDim: (BSZ_N,BSZ_M, 1) - 2D thread block with BSZ_N*BSZ_M threads
 * - gridDim: (CEIL_DIV(N, BSZ_N), CEIL_DIV(M, BSZ_M), 1)
 *
 *
 * Performs the operation: C = alpha * A * B + beta * C
 */

template <const int BSZ_M, const int BSZ_K, const int BSZ_N>
__global__ void sgemm_blocktiling(const float *A, const float *B, float *C,
                                  int M, int K, int N, float alpha,
                                  float beta) {
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    const int block_start_row = blockIdx.y * BSZ_M;
    const int block_start_col = blockIdx.x * BSZ_N;

    const int global_row = block_start_row + thread_row;
    const int global_col = block_start_col + thread_col;

    const int thread_id = thread_row * BSZ_N + thread_col;
    const int nthreads_block = BSZ_M * BSZ_N;

    __shared__ float As[BSZ_M][BSZ_K];
    __shared__ float Bs[BSZ_K][BSZ_N];

    float dot = 0.0f;

    for (int block_tile_start = 0; block_tile_start < K;
         block_tile_start += BSZ_K) {

        // strided loads
        for (int i = thread_id; i < (BSZ_M * BSZ_K); i += nthreads_block) {
            int r = i / BSZ_K;
            int c = i % BSZ_K;
            int g_r = block_start_row + r;
            int g_c = block_tile_start + c;

            As[r][c] = (g_r < M && g_c < K) ? A[g_r * K + g_c] : 0.0f;
        }

        for (int i = thread_id; i < (BSZ_K * BSZ_N); i += nthreads_block) {
            int r = i / BSZ_N;
            int c = i % BSZ_N;
            int g_r = block_tile_start + r;
            int g_c = block_start_col + c;

            Bs[r][c] = (g_r < K && g_c < N) ? B[g_r * N + g_c] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BSZ_K; ++k) {
            dot += As[thread_row][k] * Bs[k][thread_col];
        }

        __syncthreads();
    }

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] =
            alpha * dot + beta * C[global_row * N + global_col];
    }
}

/**
 * @brief Block-tiled HGEMM kernel using shared memory for improved performance
 * @tparam BSZ_M Block size in M dimension (rows processed per thread block)
 * @tparam BSZ_K Block size in K dimension (shared dimension tile size)
 * @tparam BSZ_N Block size in N dimension (columns processed per thread block)
 * @param A Input matrix A (M x K) in row-major order, device memory
 * @param B Input matrix B (K x N) in row-major order, device memory
 * @param C Output matrix C (M x N) in row-major order, device memory
 * @param M Number of rows in matrices A and C
 * @param K Number of columns in A and rows in B (shared dimension)
 * @param N Number of columns in matrices B and C
 * @param alpha Scalar multiplier for A*B product
 * @param beta Scalar multiplier for existing C values
 *
 * This kernel uses block tiling with shared memory to improve data reuse and
 * reduce global memory bandwidth requirements. Each thread block computes a
 * BSZ_M x BSZ_N tile of the output matrix C by processing the shared K
 * dimension in chunks of size BSZ_K.
 *
 * Launch configuration requirements:
 * - blockDim: (BSZ_N,BSZ_M, 1) - 2D thread block with BSZ_N*BSZ_M threads
 * - gridDim: (CEIL_DIV(N, BSZ_N), CEIL_DIV(M, BSZ_M), 1)
 *
 *
 * Performs the operation: C = alpha * A * B + beta * C
 */

template <const int BSZ_M, const int BSZ_K, const int BSZ_N>
__global__ void hgemm_blocktiling(const half *A, const half *B, half *C, int M,
                                  int K, int N, float alpha, float beta) {
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    const int block_start_row = blockIdx.y * BSZ_M;
    const int block_start_col = blockIdx.x * BSZ_N;

    const int global_row = block_start_row + thread_row;
    const int global_col = block_start_col + thread_col;

    const int thread_id = thread_row * BSZ_N + thread_col;
    const int nthreads_block = BSZ_M * BSZ_N;

    __shared__ half As[BSZ_M][BSZ_K];
    __shared__ half Bs[BSZ_K][BSZ_N];

    float dot = 0.0f;

    for (int block_tile_start = 0; block_tile_start < K;
         block_tile_start += BSZ_K) {

        // strided loads
        for (int i = thread_id; i < (BSZ_M * BSZ_K); i += nthreads_block) {
            int r = i / BSZ_K;
            int c = i % BSZ_K;
            int g_r = block_start_row + r;
            int g_c = block_tile_start + c;

            As[r][c] = (g_r < M && g_c < K) ? A[g_r * K + g_c] : (half)0.0f;
        }

        for (int i = thread_id; i < (BSZ_K * BSZ_N); i += nthreads_block) {
            int r = i / BSZ_N;
            int c = i % BSZ_N;
            int g_r = block_tile_start + r;
            int g_c = block_start_col + c;

            Bs[r][c] = (g_r < K && g_c < N) ? B[g_r * N + g_c] : (half)0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BSZ_K; ++k) {
            dot += __half2float(As[thread_row][k] * Bs[k][thread_col]);
        }

        __syncthreads();
    }

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = __float2half(
            alpha * dot + beta * __half2float(C[global_row * N + global_col]));
    }
}