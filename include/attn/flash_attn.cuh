#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "utils.cuh"

#define INF __int_as_float(0xff800000)


template <const uint BSZ_R, const uint BSZ_C>
__global__ void flash_attn(const float *Q, const float *K, const float *V,
                           float *O, int N, int d) {

    __shared__ float Qs[BSZ_R][d];
    __shared__ float Ks[BSZ_C][d];
    __shared__ float Vs[BSZ_C][d];

    const uint thread_row = threadIdx.y;
    const uint global_row = blockIdx.x * BSZ_R + thread_row;
    const uint num_col_blocks = CEIL_DIV(N, BSZ_C);

    for (int k = threadIdx.x; k < d; k += BSZ_C) {
        Qs[thread_row][k] = (global_row < N) ? Q[global_row * d + k] : 0.0f;
    }

    __syncthreads();

    float m_i = -INF;
    float l_i = 0.0f;
    float o_acc[d] = {0.0f};

    for (int j = 0; j < num_col_blocks; ++j) {

        for (int k = threadIdx.x; k < d; k += blockDim.x) {
            int global_col_k = j * BSZ_C + threadIdx.y;
            if (global_col_k < N) {
                Ks[threadIdx.y][k] = K[global_col_k * d + k];
                Vs[threadIdx.y][k] = V[global_col_k * d + k];
            } else {
                Ks[threadIdx.y][k] = 0.0f;
                Vs[threadIdx.y][k] = 0.0f;
            }
        }
        __syncthreads();

        float s_ij = 0.0f;
        for (int k = 0; k < d; ++k) {
            s_ij += Qs[thread_row][k] * Ks[threadIdx.x][k];
        }

        __shared__ float s_row_max[BSZ_R];
        if (threadIdx.x == 0)
            s_row_max[thread_row] = -INF;
    
        __syncthreads();
        atomicMax(&s_row_max[thread_row], s_ij);
        __syncthreads();

        float m_ij = s_row_max[thread_row];


        float m_new = fmaxf(m_i, m_ij);
        float p_ij_unnormalized = expf(s_ij - m_new);

        __shared__ float s_row_sum[BSZ_R];
        if (threadIdx.x == 0)
            s_row_sum[thread_row] = 0.0f;

        __syncthreads();
        atomicAdd(&s_row_sum[thread_row], p_ij_unnormalized);
        __syncthreads();

        float l_ij = s_row_sum[thread_row];

        float l_new = expf(m_i - m_new) * l_i + l_ij;


        float scale = expf(m_i - m_new) * l_i / l_new;
        for (int k = 0; k < d; ++k) {
            o_acc[k] *= scale;
        }

        for (int k = 0; k < d; ++k) {
            o_acc[k] += (p_ij_unnormalized / l_new) * Vs[threadIdx.x][k];
        }

        m_i = m_new;
        l_i = l_new;

        __syncthreads();
    }

    if (global_row < N) {
        for (int k = 0; k < d; ++k) {
            O[global_row * d + k] = o_acc[k];
        }
    }
}