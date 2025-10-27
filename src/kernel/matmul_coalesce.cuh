#pragma once

#include <driver_types.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

/*
Global Memory Coalescing matmul kernel.
Assumes blockDim(BLOCK_SZ*BLOCK_SZ,1,1),gridDim(CEIL_DIV(M,BLOCK_SZ),CEIL_DIV(N,BLOCK_SZ),1).
Assumes row-major arrays.
A: M x K
B: K x N
C: M x N
*/
template <const int BLOCK_SZ>
__global__ void matmul_coalesce(const float *A,
                                const float *B,
                                float *C,
                                int M, int K, int N)
{
    const uint trow = threadIdx.x / BLOCK_SZ;
    const uint tcol = threadIdx.x % BLOCK_SZ;

    const uint crow = blockIdx.y * BLOCK_SZ + trow;
    const uint ccol = blockIdx.x * BLOCK_SZ + tcol;

    if(crow < M && ccol < N){
        double dot = 0.0;
        for(uint k = 0; k < K; ++k){
            dot += A[crow*K + k]*B[k*N + ccol];
        }
        C[crow*N + ccol] = (float)dot;
    }
}