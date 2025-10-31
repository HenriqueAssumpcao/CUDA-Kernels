#pragma once

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <driver_types.h>

#include "attn.cuh"
#include "runners_sgemm.cuh"
#include "utils.cuh"

template <const uint BSZ, const uint TSZ_N>
__host__ void run_transpose(const float *A, float *At, int M, int N) {

    dim3 block_dim(BSZ * (BSZ / TSZ_N), 1, 1);
    dim3 grid_dim(CEIL_DIV(N, BSZ), CEIL_DIV(M, BSZ), 1);

    transpose<BSZ, TSZ_N><<<grid_dim, block_dim>>>(A, At, M, N);
}

template <const uint BSZ>
__host__ void run_softmax(const float *scores, float *attn_scores, int N, int d) {

    dim3 block_dim(BSZ, 1, 1);
    dim3 grid_dim(1, CEIL_DIV(N, BSZ), 1);

    softmax<BSZ><<<grid_dim, block_dim>>>(scores, attn_scores, N, d);
}

__host__ void run_attn_naive(const float *Q, const float *K, const float *V,
                             float *attn_scores, float *output, int N, int d,
                             float attn_scaling) {

    float *Kt, *S;
    CUDA_CHECK_ERROR(cudaMalloc(&Kt, d * N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&S, N * N * sizeof(float)));

    // tranpose K
    run_transpose<32, 8>(K, Kt, N, d);

    // attn scores: matmul + sm
    run_sgemm_threadtiling<64, 8, 64, 4, 4>(Q, Kt, S, N, d, N, attn_scaling, 0);
    run_softmax<32>(S, attn_scores, N, d);

    // out: matmul
    run_sgemm_threadtiling<64, 8, 64, 4, 4>(attn_scores, V, output, N, N, d, 1,
                                            0);

    CUDA_CHECK_ERROR(cudaFree(Kt));
    CUDA_CHECK_ERROR(cudaFree(S));
}

template <const uint BSZ_R, const uint BSZ_C>
__host__ void run_flash_attn(const float *Q, const float *K, const float *V,
                           float *O, int N, int d)
{
    dim3 block_dim(BSZ_C,BSZ_R);
    dim3 grid_dim(CEIL_DIV(N, BSZ_R),1,1);
    flash_attn<BSZ_R,BSZ_C><<<grid_dim,block_dim>>>(Q,K,V,O,N,d);
}