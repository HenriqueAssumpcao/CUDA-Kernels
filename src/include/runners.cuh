#pragma once

#include <driver_types.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "kernel.cuh"

#define CEIL_DIV(x, y) (x + y - 1) / y

template <const uint BLOCK_SZ>
__host__ void run_matmul_naive(const float *A,
                             const float *B,
                             float *C,
                             int M, int K, int N)
{
    dim3 gridDim(CEIL_DIV(M,BLOCK_SZ),CEIL_DIV(N,BLOCK_SZ),1);
    dim3 blockDim(BLOCK_SZ,BLOCK_SZ,1);

    matmul_naive<BLOCK_SZ><<<gridDim,blockDim>>>(A,B,C,M,K,N);
}

template <const uint BLOCK_SZ>
__host__ void run_matmul_naive_ref(const float *A,
                             const float *B,
                             float *C,
                             int M, int K, int N)
{
    dim3 gridDim(CEIL_DIV(M,BLOCK_SZ),CEIL_DIV(N,BLOCK_SZ),1);
    dim3 blockDim(BLOCK_SZ,BLOCK_SZ,1);

    sgemm_naive_ref<<<gridDim,blockDim>>>(A,B,C,M,K,N);
}

template <const uint BLOCK_SZ>
__host__ void run_matmul_coalesce(const float *A,
                             const float *B,
                             float *C,
                             int M, int K, int N)
{
    dim3 gridDim(CEIL_DIV(M,BLOCK_SZ),CEIL_DIV(N,BLOCK_SZ),1);
    dim3 blockDim(BLOCK_SZ*BLOCK_SZ,1,1);

    matmul_coalesce<BLOCK_SZ><<<gridDim,blockDim>>>(A,B,C,M,K,N);
}

template <const uint BLOCK_SZ>
__host__ void run_matmul_coalesce_ref(const float *A,
                             const float *B,
                             float *C,
                             int M, int K, int N)
{
    dim3 gridDim(CEIL_DIV(M,BLOCK_SZ),CEIL_DIV(N,BLOCK_SZ),1);
    dim3 blockDim(BLOCK_SZ*BLOCK_SZ,1,1);

    sgemm_global_mem_coalesce<BLOCK_SZ><<<gridDim,blockDim>>>(A,B,C,M,K,N);
}

template <const uint BLOCK_SZ>
__host__ void run_matmul_shared(const float *A,
                             const float *B,
                             float *C,
                             int M, int K, int N)
{
    dim3 gridDim(CEIL_DIV(M,BLOCK_SZ),CEIL_DIV(N,BLOCK_SZ),1);
    dim3 blockDim(BLOCK_SZ*BLOCK_SZ,1,1);

    matmul_shared<BLOCK_SZ><<<gridDim,blockDim>>>(A,B,C,M,K,N);
}

template <const uint BLOCK_SZ>
__host__ void run_matmul_shared_ref(const float *A,
                             const float *B,
                             float *C,
                             int M, int K, int N)
{
    dim3 gridDim(CEIL_DIV(M,BLOCK_SZ),CEIL_DIV(N,BLOCK_SZ),1);
    dim3 blockDim(BLOCK_SZ*BLOCK_SZ,1,1);

    sgemm_shared_mem_block<BLOCK_SZ><<<gridDim,blockDim>>>(A,B,C,M,K,N);
}