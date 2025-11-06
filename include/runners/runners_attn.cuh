#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <driver_types.h>

#include "attn.cuh"
#include "runners_sgemm.cuh"
#include "utils.cuh"

/**
 * @brief Host-side runner for the transpose kernel.
 */
template <const int BSZ, const int TSZ_N>
__host__ void run_transpose(const float *A, float *At, int M, int N) {

    dim3 block_dim(BSZ * (BSZ / TSZ_N), 1, 1);
    dim3 grid_dim(CEIL_DIV(N, BSZ), CEIL_DIV(M, BSZ), 1);

    transpose<BSZ, TSZ_N><<<grid_dim, block_dim>>>(A, At, M, N);
}

/**
 * @brief Host-side runner for the softmax kernel.
 */
template <const int BSZ>
__host__ void run_softmax(const float *scores, float *attn_scores, int N,
                          int d) {

    dim3 block_dim(BSZ, 1, 1);
    dim3 grid_dim(1, CEIL_DIV(N, BSZ), 1);

    softmax<BSZ><<<grid_dim, block_dim>>>(scores, attn_scores, N, d);
}

/**
 * @brief Host-side runner for naive attention kernel.
 */
__host__ void run_naive_attn_fwd(const float *Q, const float *K, const float *V,
                                 float *O, int N, int d, float attn_scaling) {

    float *Kt, *S, *P;
    CUDA_CHECK_ERROR(cudaMalloc(&Kt, d * N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&S, N * N * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&P, N * N * sizeof(float)));

    // tranpose K
    run_transpose<32, 8>(K, Kt, N, d);

    // attn scores: matmul + sm
    run_sgemm_threadtiling<64, 8, 64, 4, 4>(Q, Kt, S, N, d, N, attn_scaling, 0);
    run_softmax<32>(S, P, N, d);

    // out: matmul
    run_sgemm_threadtiling<64, 8, 64, 4, 4>(P, V, output, N, N, d, 1, 0);

    CUDA_CHECK_ERROR(cudaFree(Kt));
    CUDA_CHECK_ERROR(cudaFree(S));
    CUDA_CHECK_ERROR(cudaFree(P));
}

/**
 * @brief Host-side runner for theFP32 FlashAttention kernel.
 */
template <const int BSZ_R, const int BSZ_C, const int BSZ_D>
__host__ void run_flash_attn_fwd_fp32(const float *Q, const float *K,
                                      const float *V, float *O, int N, int d,
                                      float attn_scaling) {
    dim3 block_dim(BSZ_C, BSZ_R);
    dim3 grid_dim(1, CEIL_DIV(N, BSZ_R), 1);
    static_assert((BSZ_C % 32) == 1, "BSZ_C must be a multiple of 32.");
    static_assert(BSZ_C <= 1024, "BSZ_C must be at most 1024.");
    flash_attn_fwd_fp32<BSZ_R, BSZ_C, BSZ_D>
        <<<grid_dim, block_dim>>>(Q, K, V, O, N, d, attn_scaling);
}

/**
 * @brief Host-side runner for the FP16 FlashAttention kernel.
 */
template <const int BSZ_R, const int BSZ_C, const int BSZ_D>
__host__ void run_flash_attn_fwd_fp16(const half *Q, const half *K,
                                      const half *V, half *O, int N, int d,
                                      float attn_scaling) {
    dim3 block_dim(BSZ_C, BSZ_R);
    dim3 grid_dim(1, CEIL_DIV(N, BSZ_R), 1);
    static_assert((BSZ_C % 32) == 0,
                  "BSZ_C must be a multiple of 32 for warp shuffles.");
    static_assert(BSZ_C <= 1024, "BSZ_C must be at most 1024.");
    flash_attn_fwd_fp16<BSZ_R, BSZ_C, BSZ_D>
        <<<grid_dim, block_dim>>>(Q, K, V, O, N, d, attn_scaling);
}

/**
 * @brief Host-side runner for the batched, multi-head FP32 FlashAttention
 * kernel.
 */
template <const int BSZ_R, const int BSZ_C, const int BSZ_D>
__host__ void run_flash_attn_fwd_fp32_bh(const float *Q, const float *K,
                                         const float *V, float *O, int B, int H,
                                         int N, int d, float attn_scaling) {

    dim3 block_dim(BSZ_C, BSZ_R);
    dim3 grid_dim(H, B, CEIL_DIV(N, BSZ_R));

    static_assert((BSZ_C % 32) == 0,
                  "BSZ_C must be a multiple of 32 for warp shuffles.");
    static_assert(BSZ_C <= 1024, "BSZ_C must be at most 1024.");

    flash_attn_fwd_mh_fp32<BSZ_R, BSZ_C, BSZ_D>
        <<<grid_dim, block_dim>>>(Q, K, V, O, B, H, N, attn_scaling);
}

/**
 * @brief Host-side runner for the batched, multi-head FP16 FlashAttention
 * kernel.
 */
template <const int BSZ_R, const int BSZ_C, const int BSZ_D>
__host__ void run_flash_attn_fwd_fp16_bh(const half *Q, const half *K,
                                         const half *V, half *O, int B, int H,
                                         int N, int d, float attn_scaling) {

    dim3 block_dim(BSZ_C, BSZ_R);
    dim3 grid_dim(H, B, CEIL_DIV(N, BSZ_R));

    static_assert((BSZ_C % 32) == 0,
                  "BSZ_C must be a multiple of 32 for warp shuffles.");
    static_assert(BSZ_C <= 1024, "BSZ_C must be at most 1024.");

    flash_attn_fwd_mh_fp16<BSZ_R, BSZ_C, BSZ_D>
        <<<grid_dim, block_dim>>>(Q, K, V, O, B, H, N, attn_scaling);
}