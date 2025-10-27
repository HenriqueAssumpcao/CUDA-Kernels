#pragma once

#include <driver_types.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

/*
Shared memory and gmc matmul kernel.
Assumes blockDim(BLOCK_SZ*BLOCK_SZ,1,1),gridDim(CEIL_DIV(M,BLOCK_SZ),CEIL_DIV(N,BLOCK_SZ),1).
Assumes row-major arrays.
A: M x K
B: K x N
C: M x N
*/
template <const int BLOCK_SZ>
__global__ void matmul_shared(const float *A,
                              const float *B,
                              float *C,
                              int M, int K, int N)
{
    const uint trow = threadIdx.x / BLOCK_SZ;
    const uint tcol = threadIdx.x % BLOCK_SZ;

    const uint crow = blockIdx.y * BLOCK_SZ + trow;
    const uint ccol = blockIdx.x * BLOCK_SZ + tcol;

    const float *A_ptr = A + blockIdx.y * BLOCK_SZ * K;
    const float *B_ptr = B + blockIdx.x * BLOCK_SZ;

    __shared__ float A_shared[BLOCK_SZ*BLOCK_SZ];
    __shared__ float B_shared[BLOCK_SZ*BLOCK_SZ];

    
    double dot = 0.0;
    for(uint bIdx = 0; bIdx < K; bIdx += BLOCK_SZ){
        A_shared[trow * BLOCK_SZ + tcol] = (crow < M && (bIdx + tcol) < K) ? A_ptr[trow * K + tcol] : 0.0f;
        B_shared[trow * BLOCK_SZ + tcol] = ((bIdx + trow) < K && ccol < N) ? B_ptr[trow * N + tcol] : 0.0f;

        __syncthreads();

        for(uint k = 0; k < BLOCK_SZ; ++k){
            dot += A_shared[trow*BLOCK_SZ + k] * B_shared[k*BLOCK_SZ + tcol];
        }

        A_ptr += BLOCK_SZ;
        B_ptr += BLOCK_SZ * N;

        __syncthreads();   

    }
    if(crow < M && ccol < N){
        C[crow * N + ccol] = (float)dot;
    }
}