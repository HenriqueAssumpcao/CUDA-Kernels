#pragma once
#include <cuda_runtime.h>

#include "utils.h"

/**
 * @brief Reduces a float value across all active threads in a warp.
 * @param val The float value each thread contributes.
 * @return The maximum value among all threads in the warp. Result is broadcast to all threads.
 */
__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * @brief Sums a float value across all active threads in a warp.
 * @param val The float value each thread contributes.
 * @return The sum of values among all threads in the warp. Result is broadcast to all threads.
 */
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief FlashAttention Forward Pass Kernel.
 *
 * @tparam BSZ_R Block size for rows of Q (queries) per thread block.
 * @tparam BSZ_C Block size for columns of K/V (keys/values) processed per main loop iteration.
 * @tparam BSZ_D Tile size for the head dimension 'd'. This allows d > BSZ_D.
 *
 * @param Q, K, V Input matrices in HBM (row-major).
 * @param O Output matrix in HBM (row-major).
 * @param N Sequence length.
 * @param d Head dimension.
 *
 * Launch configuration:
 * - gridDim: (1, CEIL_DIV(N, BSZ_R), 1)
 * - blockDim: (BSZ_C, BSZ_R, 1)  (BSZ_C must be a multiple of 32 for warp shuffles)
 */
template <const int BSZ_R, const int BSZ_C, const int BSZ_D>
__global__ void flash_attn_fwd_kernel(const float *Q,
                                      const float *K,
                                      const float *V, float *O,
                                      int N, int d, float attn_scaling) {
    
    // indexing
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    const int block_start_row = blockIdx.y * BSZ_R;

    int thread_id = thread_row * BSZ_C + thread_col;
    int nthreads_block = BSZ_C * BSZ_R;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // SMEM 
    __shared__ float Qs[BSZ_R][BSZ_D];
    __shared__ float Ks[BSZ_C][BSZ_D];
    __shared__ float Vs[BSZ_C][BSZ_D];
    __shared__ float Os[BSZ_R][BSZ_D];

    __shared__ float partial_max[BSZ_R][BSZ_C / 32];
    __shared__ float partial_sum[BSZ_R][BSZ_C / 32];

    __shared__ float m_stats[BSZ_R];
    __shared__ float l_stats[BSZ_R];
    
    // init row stats
    if (thread_col == 0) {
        for (int i = 0; i < BSZ_R; ++i) {
            m_stats[i] = NEG_INF;
            l_stats[i] = 0.0f;
        }
    }
    __syncthreads();

    // iterate over dimension d
    for (int k_base = 0; k_base < d; k_base += BSZ_D) {
        // strided init of Os
        for (int i = thread_id; i < (BSZ_R * BSZ_D); i += nthreads_block) {
            Os[i / BSZ_D][i % BSZ_D] = 0.0f;
        }
        __syncthreads();

        // loop over K and V tiles
        for (int j_start = 0; j_start < N; j_start += BSZ_C) {
            // compute S_ij
            float s_ij_reg = 0.0f;
            for (int k_inner = 0; k_inner < d; k_inner += BSZ_D) {

                // strided loads
                for (int i = thread_id; i < BSZ_R * BSZ_D; i += nthreads_block) {
                    int r = i / BSZ_D;
                    int c = i % BSZ_D;
                    int g_r = block_start_row + r;
                    int g_c = k_inner + c;
                    Qs[r][c] = (g_r < N && g_c < d) ? Q[g_r * d + g_c] : 0.0f;
                }
                for (int i = thread_id; i < BSZ_C * BSZ_D; i += nthreads_block) {
                    int r = i / BSZ_D;
                    int c = i % BSZ_D;
                    int g_r = j_start + r;
                    int g_c = k_inner + c;
                    Ks[r][c] = (g_r < N && g_c < d) ? K[g_r * d + g_c] : 0.0f;
                }
                __syncthreads();

                if (block_start_row + thread_row < N && j_start + thread_col < N) {
                    for (int k_dot = 0; k_dot < BSZ_D; ++k_dot) {
                        s_ij_reg += Qs[thread_row][k_dot] * Ks[thread_col][k_dot];
                    }
                }
                __syncthreads();
            }
            s_ij_reg *= attn_scaling;

            // warp-level reduction
            float m_ij_warp = warp_reduce_max(s_ij_reg);
            if (lane_id == 0) partial_max[thread_row][warp_id] = m_ij_warp;
            __syncthreads();

            // block_level reduction by first warp
            float m_ij = (thread_col < (BSZ_C / 32)) ? partial_max[thread_row][thread_col] : NEG_INF;
            m_ij = warp_reduce_max(m_ij);

            float m_prev = m_stats[thread_row];
            float m_new = fmaxf(m_prev, m_ij);
            if (thread_col == 0) {
                m_stats[thread_row] = m_new;
            }

            float p_ij_unnormalized = expf(s_ij_reg - m_new);
            float l_ij_warp = warp_reduce_sum(p_ij_unnormalized);
            if (lane_id == 0) partial_sum[thread_row][warp_id] = l_ij_warp;
            __syncthreads();
            
            float l_ij = (thread_col < (BSZ_C / 32)) ? partial_sum[thread_row][thread_col] : 0.0f;
            l_ij = warp_reduce_sum(l_ij);

            float l_prev = l_stats[thread_row];
            float l_new = expf(m_prev - m_new) * l_prev + l_ij;
            if (thread_col == 0) {
                l_stats[thread_row] = l_new;
            }

            // update O
            float scale_o = expf(m_prev - m_new);
            float inv_l_new = 1.0f / l_new;

            // strided load
            for (int i = thread_id; i < BSZ_C * BSZ_D; i += nthreads_block) {
                int r = i / BSZ_D;
                int c = i % BSZ_D;
                int g_r = j_start + r;
                int g_c = k_base + c;
                Vs[r][c] = (g_r < N && g_c < d) ? V[g_r * d + g_c] : 0.0f;
            }
            __syncthreads();

            // scaling
            for (int k = thread_col; k < BSZ_D; k += blockDim.x) {
                Os[thread_row][k] *= scale_o;
            }
            
            // add contributions
            if (block_start_row + thread_row < N) {
                for (int k_out = 0; k_out < BSZ_D; ++k_out) {
                    atomicAdd(&Os[thread_row][k_out], (p_ij_unnormalized * inv_l_new) * Vs[thread_col][k_out]);
                }
            }
            __syncthreads();
        }

        // write to GMEM
        int g_row = block_start_row + thread_row;
        if (g_row < N) {
            float inv_l_final = 1.0f / l_stats[thread_row];
            for (int k = thread_col; k < BSZ_D; k += blockDim.x) {
                int g_col = k_base + k;
                if (g_col < d) {
                    O[g_row * d + g_col] = Os[thread_row][k] * inv_l_final;
                }
            }
        }
        __syncthreads(); 
    }
}
