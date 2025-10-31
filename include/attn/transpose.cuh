#pragma once

#include <stdlib.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <driver_types.h>

/* 
Launch Configuration:
blockDim(BSZ * (BSZ/TSZ_N), 1, 1)
gridDim(CEIL_DIV(N, BSZ), CEIL_DIV(M, BSZ), 1)

Assumes A: M x N is stored in row-major format.
At: N x M is stored in row-major format.
*/
template <const uint BSZ, const uint TSZ_N>
__global__ void transpose(const float *A, float *At, int M, int N) {

    const uint nthreads_col = (BSZ/TSZ_N);

    const uint thread_row = threadIdx.x / nthreads_col;
    const uint thread_col = threadIdx.x % nthreads_col;

    const uint global_row = blockIdx.y * BSZ + thread_row;
    const uint global_col = blockIdx.x * BSZ + thread_col;

    __shared__ float As[BSZ * (BSZ + 1)];

    for(uint j = 0; j < BSZ; j += nthreads_col){
        As[thread_row * (BSZ + 1) + (thread_col + j)] = (global_row < M && (global_col + j) < N) ? A[global_row * N + (global_col + j)] : 0.0f;
    }

    __syncthreads();

    global_row = blockIdx.x * BSZ + thread_row;
    global_col = blockIdx.y * BSZ + thread_col;

    for(int j = 0; j < BSZ; j += nthreads_col){
        if(global_row < N && (global_col + j) < M){
            At[global_row * M + (global_col + j)] = As[(thread_col + j) * (BSZ + 1) + thread_row];
        }
    }
}