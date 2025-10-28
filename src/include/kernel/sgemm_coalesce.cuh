#pragma once

#include <stdlib.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <driver_types.h>

/*
GMEM coalescing sgemm kernel.
Assumes
blockDim(BLOCK_SZ*BLOCK_SZ,1,1),gridDim(CEIL_DIV(N,BLOCK_SZ),CEIL_DIV(M,BLOCK_SZ),1).
Assumes row-major arrays.
A: M x K
B: K x N
C: M x N
*/
template <const uint BLOCK_SZ>
__global__ void sgemm_coalesce(const float *A, const float *B, float *C, int M,
                               int K, int N, float alpha, float beta) {
    const uint thread_row = (threadIdx.x / BLOCK_SZ);
    const uint thread_col = (threadIdx.x % BLOCK_SZ);

    const uint global_row = blockIdx.y * BLOCK_SZ + thread_row;
    const uint global_col = blockIdx.x * BLOCK_SZ + thread_col;

    if (global_row < M && global_col < N) {
        float dot = 0.0f;
        for (uint k = 0; k < K; ++k) {
            dot += A[global_row * K + k] * B[k * N + global_col];
        }
        C[global_row * N + global_col] =
            alpha * dot + beta * C[global_row * N + global_col];
    }
}