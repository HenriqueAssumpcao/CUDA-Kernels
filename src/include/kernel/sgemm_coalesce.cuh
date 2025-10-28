#pragma once

#include <stdlib.h>

#include <driver_types.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

/*
GMEM coalescing sgemm kernel.
Assumes blockDim(BLOCK_SZ*BLOCK_SZ,1,1),gridDim(CEIL_DIV(N,BLOCK_SZ),CEIL_DIV(M,BLOCK_SZ),1).
Assumes row-major arrays.
A: M x K
B: K x N
C: M x N
*/
template <const uint BLOCK_SZ>
__global__ void sgemm_coalesce(const float *A,
                                const float *B,
                                float *C,
                                int M, int K, int N,
                                float alpha, float beta)
{
    const int trow = (threadIdx.x / BLOCK_SZ);
    const uint tcol = (threadIdx.x % BLOCK_SZ);

    const int crow = blockIdx.y * BLOCK_SZ + trow;
    const int ccol = blockIdx.x * BLOCK_SZ + tcol;

    if(crow < M && ccol < N){
        float dot = 0.0f;
        for(uint k = 0; k < K; ++k){
            dot += A[crow * K + k] * B[k * N + ccol];
        }
        C[crow * N + ccol] = alpha * dot + beta * C[crow * N + ccol];
    }
}