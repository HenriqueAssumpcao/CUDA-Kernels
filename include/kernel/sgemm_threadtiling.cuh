#pragma once
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <driver_types.h>

/**
 * @brief Advanced SGEMM kernel with both block tiling and thread-level tiling
 * @tparam BM Block size in M dimension (rows processed per thread block)
 * @tparam BK Block size in K dimension (shared dimension tile size)
 * @tparam BN Block size in N dimension (columns processed per thread block)
 * @tparam TM Thread tile size in M dimension (rows per thread)
 * @tparam TN Thread tile size in N dimension (columns per thread)
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
 * for maximum performance. Each thread block computes a BM x BN tile of matrix C,
 * while each thread within the block computes a TM x TN sub-tile, significantly
 * increasing the computational intensity per thread.
 * 
 * Launch configuration requirements:
 * - blockDim: ((BN/TN) * (BM/TM), 1, 1) - fewer threads since each does more work
 * - gridDim: (CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1)
 * 
 * Template parameter constraints:
 * - BK <= min(BM, BN) for proper shared memory loading
 * - BM must be divisible by TM, BN must be divisible by TN
 * 
 * Performs the operation: C = alpha * A * B + beta * C
 */
template <const uint BM, const uint BK, const uint BN, const uint TM, const uint TN>
__global__ void sgemm_threadtiling(const float *A, const float *B, float *C, int M,
                                   int K, int N, float alpha, float beta)
{
    const uint nthreads_x = (BN / TN); 
    const uint nthreads = blockDim.x;

    const uint thread_row = threadIdx.x / nthreads_x; 
    const uint thread_col = threadIdx.x % nthreads_x;

    const uint block_start_row = blockIdx.y * BM; 
    const uint block_start_col = blockIdx.x * BN;

    const uint global_row_start = block_start_row + thread_row * TM;
    const uint global_col_start = block_start_col + thread_col * TN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float dot[TM * TN] = {0.0f};
    for(uint tile_start = 0; tile_start < K; tile_start += BK){
        for (uint i = threadIdx.x; i < BM * BK; i += nthreads) {
            uint row = i / BK;
            uint col = i % BK;
            uint g_row = block_start_row + row;
            uint g_col = tile_start + col;
            if (g_row < M && g_col < K) {
                As[i] = A[g_row * K + g_col];
            } else {
                As[i] = 0.0f;
            }
        }

        for (uint i = threadIdx.x; i < BK * BN; i += nthreads) {
            uint row = i / BN;
            uint col = i % BN;
            uint g_row = tile_start + row;
            uint g_col = block_start_col + col;
            if (g_row < K && g_col < N) {
                Bs[i] = B[g_row * N + g_col];
            } else {
                Bs[i] = 0.0f;
            }
        }
        __syncthreads();
        for(uint k = 0; k < BK; ++k){
            float regA[TM];
            for(uint i = 0; i < TM; ++i){
                regA[i] = As[(thread_row * TM + i) * BK + k];
            }
            float regB[TN];
            for(uint j = 0; j < TN; ++j){
                regB[j] = Bs[k * BN + (thread_col * TN + j)];
            }
            for(uint i = 0; i < TM; ++i){
                for(uint j = 0; j < TN; ++j){
                    dot[i * TN + j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    for(uint i = 0; i < TM; ++i){
        for(uint j = 0; j < TN; ++j){
            uint global_row = global_row_start + i;
            uint global_col = global_col_start + j;
            if(global_row < M && global_col < N){
                C[global_row * N + global_col] = 
                    alpha * dot[i * TN + j] + beta * C[global_row * N + global_col];
            }
        }
    }
}