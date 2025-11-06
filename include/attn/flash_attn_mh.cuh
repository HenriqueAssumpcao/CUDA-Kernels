#pragma once
#include "utils.cuh"
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/**
 * @brief Single precision FlashAttention Forward Pass Kernel with multiple
 * heads and batches.
 *
 * @tparam BSZ_R Block size for rows of Q per thread block.
 * @tparam BSZ_C Block size for rows of K/V processed per main loop iteration.
 * @tparam D_HEAD Size of column dimension of input matrices.
 *
 * @param Q, K, V Input matrices in HBM (row-major, layout B, H, N, D_HEAD).
 * @param O Output matrix in HBM (row-major, layout B, H, N, D_HEAD).
 * @param B Batch size.
 * @param H Number of heads.
 * @param N Sequence length.
 * @param d Head dimension.
 *
 * Launch configuration:
 * - gridDim: (H, B, CEIL_DIV(N, BSZ_R))
 * - blockDim: (BSZ_C, BSZ_R, 1)
 */
template <const int BSZ_R, const int BSZ_C, const int D_HEAD>
__global__ void flash_attn_mh_fwd_fp32(const float *Q, const float *K,
                                       const float *V, float *O, int B, int H,
                                       int N, float attn_scaling) {
    // indexing
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    const int block_start_row = blockIdx.z * BSZ_R;

    const int thread_id = thread_row * BSZ_C + thread_col;
    const int nthreads_block = BSZ_R * BSZ_C;

    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    const int nwarps_block_col = BSZ_C / 32;

    // strides and offsets
    const int h_idx = blockIdx.x;
    const int b_idx = blockIdx.y;
    const long long single_head_stride = (long long)N * D_HEAD;
    const long long batch_stride = (long long)H * single_head_stride;
    const long long base_offset =
        (long long)b_idx * batch_stride + (long long)h_idx * single_head_stride;

    const float *Q_base = Q + base_offset;
    const float *K_base = K + base_offset;
    const float *V_base = V + base_offset;
    float *O_base = O + base_offset;

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

        Qs[r][c] = (g_r < N) ? Q_base[g_r * D_HEAD + c] : 0.0f;
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

            Ks[r][c] = (g_r < N) ? K_base[g_r * D_HEAD + c] : 0.0f;
            Vs[r][c] = (g_r < N) ? V_base[g_r * D_HEAD + c] : 0.0f;
        }
        __syncthreads();

        // compute dot product corresponding to current thread
        float s_ij = 0.0f; //
        for (int k = 0; k < D_HEAD; ++k) {
            s_ij = fmaf(Qs[thread_row][k], Ks[thread_col][k], s_ij);
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

        float exp_oldnew = expf(m_old - m_new);
        float exp_currnew = expf(m_ij - m_new);

        float l_old = rowsum[thread_row];
        float l_new = exp_oldnew * l_old + exp_currnew * l_ij;

        // scale Os_acc
        float inv_l_new = 1.0f / l_new;
        float o_scaling = inv_l_new * l_old * exp_oldnew;
        float pv_scaling = inv_l_new * exp_currnew;

        for (int c = thread_col; c < D_HEAD; c += BSZ_C) {
            Os_acc[thread_row][c] *= o_scaling;
        }
        __syncthreads();

        // add P_ijV_j to Os_acc using warp and block reductions
        for (int c = 0; c < D_HEAD; ++c) {
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
            if (thread_col == 0) {
                Os_acc[thread_row][c] =
                    fmaf(pv_scaling, pv_rc, Os_acc[thread_row][c]);
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
            O_base[g_r * D_HEAD + c] = Os_acc[r][c];
        }
    }
}

/**
 * @brief Mixed precision FlashAttention Forward Pass Kernel with multiple heads
 * and batches.
 *
 * @tparam BSZ_R Block size for rows of Q per thread block.
 * @tparam BSZ_C Block size for rows of K/V processed per main loop iteration.
 * @tparam D_HEAD Size of column dimension of input matrices.
 *
 * @param Q, K, V Input matrices in HBM (row-major, layout B, H, N, D_HEAD).
 * @param O Output matrix in HBM (row-major, layout B, H, N, D_HEAD).
 * @param B Batch size.
 * @param H Number of heads.
 * @param N Sequence length.
 * @param d Head dimension.
 *
 * Launch configuration:
 * - gridDim: (H, B, CEIL_DIV(N, BSZ_R))
 * - blockDim: (BSZ_C, BSZ_R, 1)
 */
template <const int BSZ_R, const int BSZ_C, const int D_HEAD>
__global__ void flash_attn_mh_fwd_fp16(const half *Q, const half *K,
                                       const half *V, half *O, int B, int H,
                                       int N, float attn_scaling) {
    // indexing
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    const int block_start_row = blockIdx.z * BSZ_R;

    const int thread_id = thread_row * BSZ_C + thread_col;
    const int nthreads_block = BSZ_R * BSZ_C;

    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    const int nwarps_block_col = BSZ_C / 32;

    // strides and offsets
    const int h_idx = blockIdx.x;
    const int b_idx = blockIdx.y;
    const long long single_head_stride = (long long)N * D_HEAD;
    const long long batch_stride = (long long)H * single_head_stride;
    const long long base_offset =
        (long long)b_idx * batch_stride + (long long)h_idx * single_head_stride;

    const half *Q_base = Q + base_offset;
    const half *K_base = K + base_offset;
    const half *V_base = V + base_offset;
    half *O_base = O + base_offset;

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
        Qs[r][c] = (g_r < N) ? Q_base[g_r * D_HEAD + c] : __float2half(0.0f);
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
                (g_r < N) ? K_base[g_r * D_HEAD + c] : __float2half(0.0f);
            Vs[r][c] =
                (g_r < N) ? V_base[g_r * D_HEAD + c] : __float2half(0.0f);
        }
        __syncthreads();

        // compute dot product corresponding to current thread
        float s_ij = 0.0f; //
        for (int k = 0; k < D_HEAD; ++k) {
            s_ij = fmaf(__half2float(Qs[thread_row][k]),
                        __half2float(Ks[thread_col][k]), s_ij);
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

        float exp_oldnew = expf(m_old - m_new);
        float exp_currnew = expf(m_ij - m_new);

        float l_old = rowsum[thread_row];
        float l_new = exp_oldnew * l_old + exp_currnew * l_ij;

        // scale Os_acc
        float inv_l_new = 1.0f / l_new;
        float o_scaling = inv_l_new * l_old * exp_oldnew;
        float pv_scaling = inv_l_new * exp_currnew;

        for (int c = thread_col; c < D_HEAD; c += BSZ_C) {
            Os_acc[thread_row][c] *= o_scaling;
        }
        __syncthreads();

        // add P_ijV_j to Os_acc using warp and block reductions
        for (int c = 0; c < D_HEAD; ++c) {
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
            if (thread_col == 0) {
                Os_acc[thread_row][c] =
                    fmaf(pv_scaling, pv_rc, Os_acc[thread_row][c]);
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
            O_base[g_r * D_HEAD + c] = __float2half(Os_acc[r][c]);
        }
    }
}