#pragma once

#include <stdlib.h>

#include <driver_types.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

/*
Shared memory sgemm kernel, where at each iteration we load a BM x BK chunk of A and BK x BN of B into SMEM.
Assumes blockDim(BM*BN,1,1),gridDim(CEIL_DIV(N,BN),CEIL_DIV(M,BM),1).
Assumes row-major arrays.
A: M x K
B: K x N
C: M x N
*/
template <const uint BM, const uint BK, const uint BN>
__global__ void sgemm_shared(const float *A,
                             const float *B,
                             float *C,
                             int M, int K, int N,
                             float alpha, float beta)
{
    const uint trow = threadIdx.x / BN;
    const uint tcol = threadIdx.x % BN;

    const uint crow = blockIdx.y * BM + trow;
    const uint ccol = blockIdx.x * BN + tcol;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN]; 

    float dot = 0.0f;
    for(uint tile_offset = 0; tile_offset < K; tile_offset += BK){
        
        // loading one BM x BN block at a time of A and B into SMEM
        for(uint kcol = 0; kcol < BK; kcol += BN){
            const uint acol = kcol + tcol;
            if(acol < BK){
                As[trow][acol] = (crow < M && (tile_offset + acol) < K) ? A[crow * K + (tile_offset + acol)] : 0.0f;
            }
        }
        for(uint krow = 0; krow < BK; krow += BM){
            const uint brow = krow + trow;
            if(brow < BK){
                Bs[brow][tcol] = ((tile_offset + brow) < K && ccol < N) ? B[(tile_offset + brow) * N + ccol] : 0.0f;
            }
        }
        __syncthreads();

        for(uint k = 0; k < BK; ++k){
            dot += As[trow][k] * Bs[k][tcol];
        }

        __syncthreads();
    }

    if(crow < M && ccol < N){
        C[crow * N + ccol] = alpha * dot + beta * C[crow * N + ccol];
    }
}
