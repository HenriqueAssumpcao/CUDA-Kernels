#pragma once

#include <stdlib.h>

#include "device_launch_parameters.h"
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
 * - blockDim: (BSZ_M,BSZ_N, 1) - 2D thread block with BSZ_M*BSZ_N threads
 * - gridDim: (CEIL_DIV(N, BSZ_N), CEIL_DIV(M, BSZ_M), 1)
 *
 * Template parameter constraints:
 * - BSZ_K <= min(BSZ_M, BSZ_N) to ensure proper shared memory loading
 *
 * Performs the operation: C = alpha * A * B + beta * C
 */

template <const int BSZ_M, const int BSZ_K, const int BSZ_N>
__global__ void sgemm_blocktiling(const float *A, const float *B, float *C,
                                  int M, int K, int N, float alpha,
                                  float beta) {
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    const int global_row = blockIdx.y * BSZ_M + thread_row;
    const int global_col = blockIdx.x * BSZ_N + thread_col;

    __shared__ float As[BSZ_M][BSZ_K];
    __shared__ float Bs[BSZ_K][BSZ_N];

    float dot = 0.0f;
    for (int btile_start = 0; btile_start < K;
         btile_start += BSZ_K) {

        if (thread_col < BSZ_K) {
            As[thread_row][thread_col] =
                (global_row < M && (btile_start + thread_col) < K)
                    ? A[global_row * K + (btile_start + thread_col)]
                    : 0.0f;
        }
        if (thread_row < BSZ_K) {
            Bs[thread_row][thread_col] =
                ((btile_start + thread_row) < K && global_col < N)
                    ? B[(btile_start + thread_row) * N + global_col]
                    : 0.0f;
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