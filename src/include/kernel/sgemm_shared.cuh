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
__global__ void sgemm_shared(const float *A, const float *B, float *C, int M,
                             int K, int N, float alpha, float beta) {
    const uint thread_row = threadIdx.x / BN;
    const uint thread_col = threadIdx.x % BN;

    const uint global_row = blockIdx.y * BM + thread_row;
    const uint global_col = blockIdx.x * BN + thread_col;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float dot = 0.0f;
    for (uint tile_start = 0; tile_start < K; tile_start += BK) {

        if (thread_col < BK) {
            As[thread_row * BK + thread_col] =
                (global_row < M && (tile_start + thread_col) < K)
                    ? A[global_row * K + (tile_start + thread_col)]
                    : 0.0f;
        }
        if (thread_row < BK) {
            Bs[thread_row * BN + thread_col] =
                ((tile_start + thread_row) < K && global_col < N)
                    ? B[(tile_start + thread_row) * N + global_col]
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
