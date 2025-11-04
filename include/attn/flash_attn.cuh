#pragma once
#include "utils.h"
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/**
 * @brief Reduces a single precision float value to the maximum across all
 * active threads in a warp. The result is broadcast back to all threads in the
 * warp.
 */
__device__ inline float warp_sreduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * @brief Reduces a single precision float value by summing across all active
 * threads in a warp. The result is broadcast back to all threads in the warp.
 */
__device__ inline float warp_sreduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Single precision FlashAttention Forward Pass Kernel.
 *
 * @tparam BSZ_R Block size for rows of Q per thread block.
 * @tparam BSZ_C Block size for rows of K/V processed per main loop iteration.
 * @tparam D_HEAD Size of column dimension of input matrices.
 *
 * @param Q, K, V Input matrices in HBM (row-major).
 * @param O Output matrix in HBM (row-major).
 * @param N Sequence length.
 * @param d Head dimension.
 *
 * Launch configuration:
 * - gridDim: (1, CEIL_DIV(N, BSZ_R), 1)
 * - blockDim: (BSZ_C, BSZ_R, 1)  (BSZ_C must be a multiple of 32 for warp
 * shuffles and BSZ_C <= 1024)
 */
template <const int BSZ_R, const int BSZ_C, const int D_HEAD>
__global__ void flash_attn_fwd_fp32(const float *Q, const float *K,
                                    const float *V, float *O, int N,
                                    float attn_scaling) {
    // indexing
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    const int block_start_row = blockIdx.y * BSZ_R;
    const int block_start_col = blockIdx.x * BSZ_C;

    const int thread_id = thread_row * BSZ_C + thread_col;
    const int nthreads_block = BSZ_R * BSZ_C;

    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    const int nwarps_block_col = BSZ_C / 32;

    // each block will be responsible for a row-block of Q and O
    __shared__ float Qs[BSZ_R][D_HEAD];
    __shared__ float Os_acc[BSZ_R][D_HEAD];

    // each block will iterate over all row-blocks of K and V
    __shared__ float Ks[BSZ_C][D_HEAD];
    __shared__ float Vs[BSZ_C][D_HEAD];

    // initialize rowmax and rowsum for each row of the block
    __shared__ float rowmax[BSZ_R];
    __shared__ float rowsum[BSZ_R];

    if (thread_col == 0) {
        rowmax[thread_row] = NEG_INF;
        rowsum[thread_row] = 0.0f;
    }

    // shared array for warp-level reductions
    __shared__ float row_warp_reduction[BSZ_R][nwarps_block_col];

    // load row-block of Q into SMEM using strided load, init Os_acc as 0
    for (int i = thread_id; i < (BSZ_R * D_HEAD); i += nthreads_block) {
        int r = i / D_HEAD;
        int c = i % D_HEAD;
        int g_r = block_start_row + r;

        Qs[r][c] = (g_r < N) ? Q[g_r * D_HEAD + c] : 0.0f;
        Os_acc[r][c] = 0.0f;
    }

    __syncthreads();

    // loop over column tiles
    for (int block_tile_start = 0; block_tile_start < N;
         block_tile_start += BSZ_C) {
        // load row-block of K and V into SMEM using strided load
        for (int i = thread_id; i < (BSZ_C * D_HEAD); i += nthreads_block) {
            int r = i / D_HEAD;
            int c = i % D_HEAD;
            int g_r = block_tile_start + r;

            Ks[r][c] = (g_r < N) ? K[g_r * D_HEAD + c] : 0.0f;
            Vs[r][c] = (g_r < N) ? V[g_r * D_HEAD + c] : 0.0f;
        }
        __syncthreads();

        // compute dot product corresponding to current thread
        float s_ij = 0.0f;
        for (int k = 0; k < D_HEAD; ++k) {
            s_ij += Qs[thread_row][k] * Ks[thread_col][k];
        }
        s_ij *= attn_scaling;

        // warp-level reduction to compute new row max for each warp
        float m_ij_warp = warp_reduce_max(s_ij);
        if (lane_id == 0) {
            row_warp_reduction[thread_row][warp_id] = m_ij_warp;
        }
        __syncthreads();

        // block-level reduction to compute new row max
        float m_ij = (thread_col < nwarps_block_col)
                         ? row_warp_reduction[thread_row][thread_col]
                         : NEG_INF;
        m_ij = warp_reduce_max(m_ij);

        // compute exp
        float p_ij = 0.0f;
        p_ij = expf(s_ij - m_ij);

        // warp-level reduction to compute new row sum for each warp
        float l_ij_warp = warp_reduce_sum(p_ij);
        if (lane_id == 0) {
            row_warp_reduction[thread_row][warp_id] = l_ij_warp;
        }
        __syncthreads();

        // block-level reduction to compute new row max
        float l_ij = (thread_col < nwarps_block_col)
                         ? row_warp_reduction[thread_row][thread_col]
                         : 0.0f;
        l_ij = warp_reduce_sum(l_ij);

        // compute new stats
        float m_old = rowmax[thread_row];
        float m_new = fmaxf(m_old, m_ij);

        float exp_oldnew = fexpf(m_old - m_new);
        float exp_currnew = fexpf(m_ij - m_new);

        float l_old = rowsum[thread_row];
        float l_new = exp_oldnew * l_old + exp_currnew * l_ij;

        // scale Os_acc
        float inv_l_new = 1.0f / l_new;
        float o_scaling = inv_l_new * l_old * exp_oldnew;
        float pv_scaling = inv_l_new * exp_currnew;

        // add P_ijV_j to Os_acc using warp and block reductions
        for (int c = thread_col; c < D_HEAD; c += BSZ_C) {
            float pv_rc_warp = p_ij * Vs[thread_col][c];
            pv_rc_warp = warp_reduce_sum(pv_rc_warp);
            if (lane_id == 0) {
                row_warp_reduction[thread_row][warp_id] = pv_rc_warp;
            }
            __syncthreads();

            float pv_rc = (thread_col < nwarps_block_col)
                              ? row_warp_reduction[thread_row][thread_col]
                              : 0.0f;
            pv_rc = warp_reduce_sum(pv_rc);

            if (thread_col == (c % BSZ_C)) {
                Os_acc[thread_row][c] =
                    o_scaling * Os_acc[thread_row][c] + pv_scaling * pv_rc;
            }
        }

        // update stats
        if (thread_col == 0) {
            rowmax[thread_row] = m_new;
            rowsum[thread_row] = l_new;
        }
        __syncthreads();
    }

    // strided save of the respective row-block of O to GMEM
    for (int i = thread_id; i < (BSZ_R * D_HEAD); i += nthreads_block) {
        int r = i / D_HEAD;
        int c = i % D_HEAD;
        int g_r = block_start_row + r;
        if (g_r < N) {
            O[g_r * D_HEAD + c] = Os_acc[r][c];
        }
    }
}

/**
 * @brief Mixed precision FlashAttention Forward Pass Kernel.
 *
 * @tparam BSZ_R Block size for rows of Q per thread block.
 * @tparam BSZ_C Block size for rows of K/V processed per main loop iteration.
 * @tparam D_HEAD Size of column dimension of input matrices.
 *
 * @param Q, K, V Input matrices in HBM (row-major, half precision).
 * @param O Output matrix in HBM (row-major, half precision).
 * @param N Sequence length.
 * @param d Head dimension.
 *
 * Launch configuration:
 * - gridDim: (1, CEIL_DIV(N, BSZ_R), 1)
 * - blockDim: (BSZ_C, BSZ_R, 1)  (BSZ_C must be a multiple of 32 for warp
 * shuffles and BSZ_C <= 1024)
 */
template <const int BSZ_R, const int BSZ_C, const int D_HEAD>
__global__ void flash_attn_fwd_fp16(const half *Q, const half *K, const half *V,
                                    half *O, int N, float attn_scaling) {
    // indexing
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    const int block_start_row = blockIdx.y * BSZ_R;
    const int block_start_col = blockIdx.x * BSZ_C;

    const int thread_id = thread_row * BSZ_C + thread_col;
    const int nthreads_block = BSZ_R * BSZ_C;

    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    const int nwarps_block_col = BSZ_C / 32;

    // each block will be responsible for a row-block of Q and O
    __shared__ half Qs[BSZ_R][D_HEAD];
    __shared__ float Os_acc[BSZ_R][D_HEAD];

    // each block will iterate over all row-blocks of K and V
    __shared__ half Ks[BSZ_C][D_HEAD];
    __shared__ half Vs[BSZ_C][D_HEAD];

    // initialize rowmax and rowsum for each row of the block
    __shared__ float rowmax[BSZ_R];
    __shared__ float rowsum[BSZ_R];

    if (thread_col == 0) {
        rowmax[thread_row] = NEG_INF;
        rowsum[thread_row] = 0.0f;
    }
    // shared array for warp-level reductions
    __shared__ float row_warp_reduction[BSZ_R][nwarps_block_col];

    // load row-block of Q into SMEM using strided load, init Os_acc as 0
    for (int i = thread_id; i < (BSZ_R * D_HEAD); i += nthreads_block) {
        int r = i / D_HEAD;
        int c = i % D_HEAD;
        int g_r = block_start_row + r;
        Qs[r][c] = (g_r < N) ? Q[g_r * D_HEAD + c] : static_cast<half>(0.0f);
        Os_acc[r][c] = 0.0f;
    }
    __syncthreads();

    // loop over column tiles
    for (int block_tile_start = 0; block_tile_start < N;
         block_tile_start += BSZ_C) {
        // load row-block of K and V into SMEM using strided load
        for (int i = thread_id; i < (BSZ_C * D_HEAD); i += nthreads_block) {
            int r = i / D_HEAD;
            int c = i % D_HEAD;
            int g_r = block_tile_start + r;

            Ks[r][c] =
                (g_r < N) ? K[g_r * D_HEAD + c] : static_cast<half>(0.0f);
            Vs[r][c] =
                (g_r < N) ? V[g_r * D_HEAD + c] : static_cast<half>(0.0f);
        }
        __syncthreads();

        // compute dot product corresponding to current thread
        float s_ij = 0.0f; //
        for (int k = 0; k < D_HEAD; ++k) {
            s_ij += __half2float(Qs[thread_row][k]) *
                    __half2float(Ks[thread_col][k]);
        }
        s_ij *= attn_scaling;

        // warp-level reduction to compute new row max for each warp
        float m_ij_warp = warp_reduce_max(s_ij);
        if (lane_id == 0) {
            row_warp_reduction[thread_row][warp_id] = m_ij_warp;
        }
        __syncthreads();

        // block-level reduction to compute new row max
        float m_ij = (thread_col < nwarps_block_col)
                         ? row_warp_reduction[thread_row][thread_col]
                         : NEG_INF;
        m_ij = warp_reduce_max(m_ij);

        // compute exp
        float p_ij = 0.0f;
        p_ij = expf(s_ij - m_ij);

        // warp-level reduction to compute new row sum for each warp
        float l_ij_warp = warp_reduce_sum(p_ij);
        if (lane_id == 0) {
            row_warp_reduction[thread_row][warp_id] = l_ij_warp;
        }
        __syncthreads();

        // block-level reduction to compute new row max
        float l_ij = (thread_col < nwarps_block_col)
                         ? row_warp_reduction[thread_row][thread_col]
                         : 0.0f;
        l_ij = warp_reduce_sum(l_ij);

        // compute new stats
        float m_old = rowmax[thread_row];
        float m_new = fmaxf(m_old, m_ij);

        float exp_oldnew = fexpf(m_old - m_new);
        float exp_currnew = fexpf(m_ij - m_new);

        float l_old = rowsum[thread_row];
        float l_new = exp_oldnew * l_old + exp_currnew * l_ij;

        // scale Os_acc
        float inv_l_new = 1.0f / l_new;
        float o_scaling = inv_l_new * l_old * exp_oldnew;
        float pv_scaling = inv_l_new * exp_currnew;

        // add P_ijV_j to Os_acc using warp and block reductions
        for (int c = thread_col; c < D_HEAD; c += BSZ_C) {
            float pv_rc_warp = p_ij * __half2float(Vs[thread_col][c]);
            pv_rc_warp = warp_reduce_sum(pv_rc_warp);
            if (lane_id == 0) {
                row_warp_reduction[thread_row][warp_id] = pv_rc_warp;
            }
            __syncthreads();

            float pv_rc = (thread_col < nwarps_block_col)
                              ? row_warp_reduction[thread_row][thread_col]
                              : 0.0f;
            pv_rc = warp_reduce_sum(pv_rc);
            if (thread_col == (c % BSZ_C)) {
                Os_acc[thread_row][c] =
                    o_scaling * Os_acc[thread_row][c] + pv_scaling * pv_rc;
            }
        }
        // update stats
        if (thread_col == 0) {
            rowmax[thread_row] = m_new;
            rowsum[thread_row] = l_new;
        }
        __syncthreads();
    }
    // strided save of the respective row-block of O to GMEM
    for (int i = thread_id; i < (BSZ_R * D_HEAD); i += nthreads_block) {
        int r = i / D_HEAD;
        int c = i % D_HEAD;
        int g_r = block_start_row + r;
        if (g_r < N) {
            O[g_r * D_HEAD + c] = __float2half(Os_acc[r][c]);
        }
    }
}