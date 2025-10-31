#pragma once
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <driver_types.h>

/**
 * @brief Advanced SGEMM kernel with both block tiling and thread-level tiling
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
 * This is the most advanced kernel combining block tiling with thread-level tiling
 * for maximum performance. Each thread block computes a BSZ_M x BSZ_N tile of matrix C,
 * while each thread within the block computes a TSZ_M x TSZ_N sub-tile, significantly
 * increasing the computational intensity per thread.
 * 
 * Launch configuration requirements:
 * - blockDim: ((BSZ_N/TSZ_N) * (BSZ_M/TSZ_M), 1, 1) - fewer threads since each does more work
 * - gridDim: (CEIL_DIV(N, BSZ_N), CEIL_DIV(M, BSZ_M), 1)
 * 
 * Template parameter constraints:
 * - BSZ_K <= min(BSZ_M, BSZ_N) for proper shared memory loading
 * - BSZ_M must be divisible by TSZ_M, BSZ_N must be divisible by TSZ_N
 * 
 * Performs the operation: C = alpha * A * B + beta * C
 */
template <const uint BSZ_M, const uint BSZ_K, const uint BSZ_N, const uint TSZ_M, const uint TSZ_N>
__global__ void sgemm_threadtiling(const float *A, const float *B, float *C, int M,
                                   int K, int N, float alpha, float beta)
{
    const uint nthreads_col = (BSZ_N / TSZ_N); 
    const uint nthreads_block = blockDim.x;

    const uint thread_row = threadIdx.x / nthreads_col; 
    const uint thread_col = threadIdx.x % nthreads_col;

    const uint block_start_row = blockIdx.y * BSZ_M; 
    const uint block_start_col = blockIdx.x * BSZ_N;

    const uint global_row_start = block_start_row + thread_row * TSZ_M;
    const uint global_col_start = block_start_col + thread_col * TSZ_N;

    __shared__ float As[BSZ_M * BSZ_K];
    __shared__ float Bs[BSZ_K * BSZ_N];

    float dot[TSZ_M * TSZ_N] = {0.0f};
    for(uint block_tile_start = 0; block_tile_start < K; block_tile_start += BSZ_K){
        for (uint i = threadIdx.x; i < BSZ_M * BSZ_K; i += nthreads_block) {
            uint row = i / BSZ_K;
            uint col = i % BSZ_K;
            uint g_row = block_start_row + row;
            uint g_col = block_tile_start + col;
            if (g_row < M && g_col < K) {
                As[i] = A[g_row * K + g_col];
            } else {
                As[i] = 0.0f;
            }
        }

        for (uint i = threadIdx.x; i < BSZ_K * BSZ_N; i += nthreads_block) {
            uint row = i / BSZ_N;
            uint col = i % BSZ_N;
            uint g_row = block_tile_start + row;
            uint g_col = block_start_col + col;
            if (g_row < K && g_col < N) {
                Bs[i] = B[g_row * N + g_col];
            } else {
                Bs[i] = 0.0f;
            }
        }
        __syncthreads();
        for(uint k = 0; k < BSZ_K; ++k){
            float regA[TSZ_M];
            for(uint i = 0; i < TSZ_M; ++i){
                regA[i] = As[(thread_row * TSZ_M + i) * BSZ_K + k];
            }
            float regB[TSZ_N];
            for(uint j = 0; j < TSZ_N; ++j){
                regB[j] = Bs[k * BSZ_N + (thread_col * TSZ_N + j)];
            }
            for(uint i = 0; i < TSZ_M; ++i){
                for(uint j = 0; j < TSZ_N; ++j){
                    dot[i * TSZ_N + j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    for(uint i = 0; i < TSZ_M; ++i){
        for(uint j = 0; j < TSZ_N; ++j){
            uint global_row = global_row_start + i;
            uint global_col = global_col_start + j;
            if(global_row < M && global_col < N){
                C[global_row * N + global_col] = 
                    alpha * dot[i * TSZ_N + j] + beta * C[global_row * N + global_col];
            }
        }
    }
}