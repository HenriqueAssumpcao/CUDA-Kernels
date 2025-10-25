#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>

#include <driver_types.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"

/*
ROW MAJOR
A: M x K,B:K x N,C:M x N
*/

/*
SGEMM naive;
blockDim(WARP_SZ,WARP_SZ,1);
*/
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float *A, const float *B,
                            float beta, float *C) {
    
    const uint crow = blockIdx.y * blockDim.y + threadIdx.y;
    const uint ccol = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(crow < M && ccol < N){
        double dot = 0.0;
        for(uint k = 0; k < K; ++k){
            dot += A[crow*K + k]*B[k*N + ccol];
        }
        C[crow*N + ccol] = alpha*dot + beta*C[crow*N + ccol];
    }
}

/*
SGEMM with global memory coalescing;
blockDim(WARP_SZ*WARP_SZ,1,1);
*/
template <const uint WARP_SZ>
__global__ void sgemm_gmc(int M, int N, int K, float alpha,
                          const float *A, const float *B,
                          float beta, float *C) {
    
    const uint trow = threadIdx.x/WARP_SZ; // trow is the same for threads within the same warp
    const uint tcol = threadIdx.x%WARP_SZ; // tcol is different for threads within the same warp

    const uint crow = blockIdx.y * blockDim.y + trow;
    const uint ccol = blockIdx.x * blockDim.x + tcol;

    if(crow < M && ccol < N){
        double dot = 0.0;
        for(uint k = 0; k < K; ++k){
            dot += A[crow*K + k]*B[k*N + ccol];
        }
        C[crow*N + ccol] = alpha*dot + beta*C[crow*N + ccol];
    }
}

/*
SGEMM with shared memory and gmc.
*/
template <const uint WARP_SZ>
__global__ void sgemm_sm(int M, int N, int K, float alpha,
                          const float *A, const float *B,
                          float beta, float *C) {

    __shared__ float shared_A[WARP_SZ*WARP_SZ];
    __shared__ float shared_B[WARP_SZ*WARP_SZ];
    
    const uint trow = threadIdx.x/WARP_SZ; // trow is the same for threads within the same warp
    const uint tcol = threadIdx.x%WARP_SZ; // tcol is different for threads within the same warp
    
    // move pointers
    A += blockIdx.y * WARP_SZ * K; // A gets moved to the start of the row
    B += blockIdx.x * WARP_SZ; // B gets moved to the start of the column
    C += blockIdx.y * WARP_SZ * N + blockIdx.x * WARP_SZ; // C gets moved to the block

    double dot = 0.0;

    for(size_t block_idx = 0; block_idx < K; block_idx += WARP_SZ){
        shared_A[trow*WARP_SZ + tcol] = A[trow*K + tcol];
        shared_B[trow*WARP_SZ + tcol] = B[trow*N + tcol];

        __syncthreads();

        for(size_t k = 0;k < WARP_SZ; ++k){
            dot += shared_A[trow*WARP_SZ + k]*shared_B[k*WARP_SZ + tcol];
        }

        A += WARP_SZ;B += (WARP_SZ*N);

        __syncthreads();        
    }

    C[trow*WARP_SZ + tcol] = alpha * dot + beta * ;


}