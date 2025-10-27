#pragma once

#include <driver_types.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

/*
Naive matmul kernel.
Assumes blockDim(BLOCK_SZ,BLOCK_SZ,1),gridDim(CEIL_DIV(M,BLOCK_SZ),CEIL_DIV(N,BLOCK_SZ),1).
Assumes row-major arrays.
A: M x K
B: K x N
C: M x N
*/
__global__ void matmul_naive(const float *A,
                             const float *B,
                             float *C,
                             int M, int K, int N)
{
    const uint trow = threadIdx.y;
    const uint tcol = threadIdx.x;

    const uint crow = blockIdx.y * blockDim.y + trow;
    const uint ccol = blockIdx.x * blockDim.x + tcol;

    if(crow < M && ccol < N){
        double dot = 0.0;
        for(uint k = 0; k < K; ++k){
            dot += A[crow*K + k]*B[k*N + ccol];
        }
        C[crow*N + ccol] = (float)dot;
    }
}
