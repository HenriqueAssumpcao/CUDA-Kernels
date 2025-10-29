#pragma once

#include <stdlib.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <driver_types.h>

/**
 * @brief Block-tiled SGEMM kernel using shared memory for improved performance
 * @tparam BM Block size in M dimension (rows processed per thread block)
 * @tparam BK Block size in K dimension (shared dimension tile size)
 * @tparam BN Block size in N dimension (columns processed per thread block)
 * @param A Input matrix A (M x K) in row-major order, device memory
 * @param B Input matrix B (K x N) in row-major order, device memory
 * @param C Output matrix C (M x N) in row-major order, device memory
 * @param M Number of rows in matrices A and C
 * @param K Number of columns in A and rows in B (shared dimension)
 * @param N Number of columns in matrices B and C
 * @param alpha Scalar multiplier for A*B product
 * @param beta Scalar multiplier for existing C values
 * 
 * This kernel uses block tiling with shared memory to improve data reuse and reduce
 * global memory bandwidth requirements. Each thread block computes a BM x BN tile
 * of the output matrix C by processing the shared K dimension in chunks of size BK.
 * 
 * Launch configuration requirements:
 * - blockDim: (BM * BN, 1, 1) - 1D thread block with BM*BN threads
 * - gridDim: (CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1)
 * 
 * Template parameter constraints:
 * - BK <= min(BM, BN) to ensure proper shared memory loading
 * 
 * Performs the operation: C = alpha * A * B + beta * C
 */
template <const uint BM, const uint BK, const uint BN>
__global__ void sgemm_blocktiling(const float *A, const float *B, float *C, int M,
                             int K, int N, float alpha, float beta) {
    const uint thread_row = threadIdx.x / BN;
    const uint thread_col = threadIdx.x % BN;

    const uint global_row = blockIdx.y * BM + thread_row;
    const uint global_col = blockIdx.x * BN + thread_col;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float dot = 0.0f;
    for (uint block_tile_start = 0; block_tile_start < K; block_tile_start += BK) {

        if (thread_col < BK) {
            As[thread_row * BK + thread_col] =
                (global_row < M && (block_tile_start + thread_col) < K)
                    ? A[global_row * K + (block_tile_start + thread_col)]
                    : 0.0f;
        }
        if (thread_row < BK) {
            Bs[thread_row * BN + thread_col] =
                ((block_tile_start + thread_row) < K && global_col < N)
                    ? B[(block_tile_start + thread_row) * N + global_col]
                    : 0.0f;
        }
        __syncthreads();

        for (uint k = 0; k < BK; ++k) {
            dot += As[thread_row * BK + k] * Bs[k * BN + thread_col];
        }

        __syncthreads();
    }

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] =
            alpha * dot + beta * C[global_row * N + global_col];
    }
}
