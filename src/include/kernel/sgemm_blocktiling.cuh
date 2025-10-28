#pragma once

#include <stdlib.h>

#include <driver_types.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

/*
2D blocktiling sgemm kernel.
Assumes blockDim((BM/TM)*(BN/TN),1,1),gridDim(CEIL_DIV(N,BN),CEIL_DIV(M,BM),1).
Assumes row-major arrays.
A: M x K
B: K x N
C: M x N
*/
template <const uint BM, const uint BK, const uint BN, const uint TM, const uint TN>
__global__ void sgemm_blocktiling(const float *A,
                                             const float *B,
                                             float *C,
                                             int M, int K, int N,
                                             float alpha, float beta)
{
    const uint thread_tile_row = threadIdx.x / (BN / TN);
    const uint thread_tile_col = threadIdx.x % (BN / TN);

    const uint local_start_row = thread_tile_row * TM;
    const uint local_start_col = thread_tile_col * TN;

    const uint global_start_row = blockIdx.y * BM + local_start_row;
    const uint global_start_col = blockIdx.x * BN + local_start_col;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float dotblock[TM][TN] = {{0.0f}};
    const uint num_threads = blockDim.x;

    for (uint k_tile_start = 0; k_tile_start < K; k_tile_start += BK) {
        
        for (uint i = threadIdx.x; i < BM * BK; i += num_threads) {
            uint load_row = i / BK;
            uint load_col = i % BK;
            uint global_A_row = blockIdx.y * BM + load_row;
            uint global_A_col = k_tile_start + load_col;
            if (global_A_row < M && global_A_col < K) {
                As[load_row][load_col] = A[global_A_row * K + global_A_col];
            } else {
                As[load_row][load_col] = 0.0f;
            }
        }
        for (uint i = threadIdx.x; i < BK * BN; i += num_threads) {
            uint load_row = i / BN;
            uint load_col = i % BN;
            uint global_B_row = k_tile_start + load_row;
            uint global_B_col = blockIdx.x * BN + load_col;
            if (global_B_row < K && global_B_col < N) {
                Bs[load_row][load_col] = B[global_B_row * N + global_B_col];
            } else {
                Bs[load_row][load_col] = 0.0f;
            }
        }
        __syncthreads();

        for (uint k = 0; k < BK; ++k) {
            float A_reg[TM];
            for (uint m = 0; m < TM; ++m) A_reg[m] = As[local_start_row + m][k];
            
            float B_reg[TN];
            for (uint n = 0; n < TN; ++n) B_reg[n] = Bs[k][local_start_col + n];
            
            for (uint m = 0; m < TM; ++m) {
                for (uint n = 0; n < TN; ++n) {
                    dotblock[m][n] += A_reg[m] * B_reg[n];
                }
            }
        }
        __syncthreads();
    }

    for (uint m = 0; m < TM; ++m) {
        for (uint n = 0; n < TN; ++n) {
            uint final_global_row = global_start_row + m;
            uint final_global_col = global_start_col + n;
            if (final_global_row < M && final_global_col < N) {
                size_t C_idx = (size_t)final_global_row * N + final_global_col;
                C[C_idx] = alpha * dotblock[m][n] + beta * C[C_idx];
            }
        }
    }
}