#pragma once

#include <driver_types.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "kernel.cuh"

#define CEIL_DIV(x, y) (x + y - 1) / y

template <const uint BLOCK_SZ>
__host__ void run_sgemm_naive(const float *A,
                              const float *B,
                              float *C,
                              int M, int K, int N,
                              float alpha, float beta)
{
    dim3 gridDim(CEIL_DIV(N,BLOCK_SZ),CEIL_DIV(M,BLOCK_SZ),1);
    dim3 blockDim(BLOCK_SZ,BLOCK_SZ,1);

    sgemm_naive<BLOCK_SZ><<<gridDim,blockDim>>>(A,B,C,M,K,N,alpha,beta);
}

template <const uint BLOCK_SZ>
__host__ void run_sgemm_coalesce(const float *A,
                                 const float *B,
                                 float *C,
                                 int M, int K, int N,
                                 float alpha, float beta)
{
    dim3 gridDim(CEIL_DIV(N,BLOCK_SZ),CEIL_DIV(M,BLOCK_SZ),1);
    dim3 blockDim(BLOCK_SZ*BLOCK_SZ,1,1);

    sgemm_coalesce<BLOCK_SZ><<<gridDim,blockDim>>>(A,B,C,M,K,N,alpha,beta);
}

template <const uint BM, const uint BK, const uint BN>
__host__ void run_sgemm_shared(const float *A,
                             const float *B,
                             float *C,
                             int M, int K, int N,
                             float alpha, float beta)
{
    dim3 gridDim(CEIL_DIV(N,BN),CEIL_DIV(M,BM),1);
    dim3 blockDim(BM*BN,1,1);

    sgemm_shared<BM,BK,BN><<<gridDim,blockDim>>>(A,B,C,M,K,N,alpha,beta);
}

template <const uint BM, const uint BK, const uint BN, const uint TM, const uint TN>
__host__ void run_sgemm_blocktiling(const float *A,
                                             const float *B,
                                             float *C,
                                             int M, int K, int N,
                                             float alpha, float beta)
{
    dim3 gridDim(CEIL_DIV(N,BN),CEIL_DIV(M,BM),1);
    dim3 blockDim((BM*BN)/(TM * TN),1,1);

    sgemm_blocktiling<BM,BK,BN,TM,TN><<<gridDim,blockDim>>>(A,B,C,M,K,N,alpha,beta);
}