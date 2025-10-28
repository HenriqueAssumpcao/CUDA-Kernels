#pragma once

#include <stdlib.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <driver_types.h>

/*
Shared memory sgemm kernel, where at each iteration we load
a BM x BK chunk of A and BK x BN of B into SMEM.
Assumes blockDim(BM*BN,1,1),gridDim(CEIL_DIV(N,BN),CEIL_DIV(M,BM),1).
Assumes BK <= min(BM,BN).
Assumes row-major arrays. A: M x K B: K x N C: M x N.
*/
template <const uint BM, const uint BK, const uint BN>
__global__ void sgemm_shared_dbuf(const float *A, const float *B, float *C, int M,
                             int K, int N, float alpha, float beta) {
    const uint thread_row = threadIdx.x / BN;
    const uint thread_col = threadIdx.x % BN;

    const uint global_row = blockIdx.y * BM + thread_row;
    const uint global_col = blockIdx.x * BN + thread_col;

    __shared__ float As[2 * BM * BK];
    __shared__ float Bs[2 * BK * BN];

    float *As_buf0 = &As[0];
    float *As_buf1 = &As[BM * BK];

    float *Bs_buf0 = &Bs[0];
    float *Bs_buf1 = &Bs[BK * BN];

    if (thread_col < BK) {
        As_buf0[thread_row * BK + thread_col] =
            (global_row < M && thread_col < K)
                ? A[global_row * K + thread_col]
                : 0.0f;
    }
    if (thread_row < BK) {
        Bs_buf0[thread_row * BN + thread_col] =
            (thread_row < K && global_col < N)
                ? B[thread_row * N + global_col]
                : 0.0f;
    }
    __syncthreads();

    float dot = 0.0f;
    int buf_idx = 0;
    for (uint tile_start = 0; tile_start < K; tile_start += BK) {

        float *As_compute_ptr = (buf_idx == 0) ? As_buf0 : As_buf1;
        float *Bs_compute_ptr = (buf_idx == 0) ? Bs_buf0 : Bs_buf1;
        float *As_load_ptr = (buf_idx == 0) ? As_buf1 : As_buf0; 
        float *Bs_load_ptr = (buf_idx == 0) ? Bs_buf1 : Bs_buf0; 

        const uint next_tile_start = tile_start + BK;
        if (next_tile_start < K) {
            if (thread_col < BK) {
                As_load_ptr[thread_row * BK + thread_col] =
                    (global_row < M && (next_tile_start + thread_col) < K)
                        ? A[global_row * K + (next_tile_start + thread_col)]
                        : 0.0f;
            }
            if (thread_row < BK) {
                Bs_load_ptr[thread_row * BN + thread_col] =
                    ((next_tile_start + thread_row) < K && global_col < N)
                        ? B[(next_tile_start + thread_row) * N + global_col]
                        : 0.0f;
            }
        }

        for (uint k = 0; k < BK; ++k) {
            dot += As_compute_ptr[thread_row * BK + k] *
                   Bs_compute_ptr[k * BN + thread_col];
        }

        __syncthreads();
        buf_idx = 1 - buf_idx;
    }

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] =
            alpha * dot + beta * C[global_row * N + global_col];
    }
}
