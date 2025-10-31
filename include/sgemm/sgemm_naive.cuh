#pragma once

#include <stdlib.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <driver_types.h>

/**
 * @brief Memory-coalesced naive CUDA kernel for single-precision general matrix multiplication (SGEMM)
 * @tparam BSZ Block size dimension (square block conceptually, but launched as 1D)
 * @param A Input matrix A (M x K) in row-major order, device memory
 * @param B Input matrix B (K x N) in row-major order, device memory
 * @param C Output matrix C (M x N) in row-major order, device memory
 * @param M Number of rows in matrices A and C
 * @param K Number of columns in A and rows in B (shared dimension)
 * @param N Number of columns in matrices B and C
 * @param alpha Scalar multiplier for A*B product
 * @param beta Scalar multiplier for existing C values
 * 
 * Each thread computes a single element of the output matrix C.
 * 
 * Launch configuration requirements:
 * - blockDim: (BSZ * BSZ, 1, 1)
 * - gridDim: (CEIL_DIV(N, BSZ), CEIL_DIV(M, BSZ), 1)
 * 
 * Performs the operation: C = alpha * A * B + beta * C
 */
template <const uint BSZ>
__global__ void sgemm_naive(const float *A, const float *B, float *C, int M,
                               int K, int N, float alpha, float beta) {
    const uint thread_row = (threadIdx.x / BSZ);
    const uint thread_col = (threadIdx.x % BSZ);

    const uint global_row = blockIdx.y * BSZ + thread_row;
    const uint global_col = blockIdx.x * BSZ + thread_col;

    if (global_row < M && global_col < N) {
        float dot = 0.0f;
        for (uint k = 0; k < K; ++k) {
            dot += A[global_row * K + k] * B[k * N + global_col];
        }
        C[global_row * N + global_col] =
            alpha * dot + beta * C[global_row * N + global_col];
    }
}