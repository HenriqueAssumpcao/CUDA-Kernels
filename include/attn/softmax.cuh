#pragma once

#include <stdlib.h>

#include <cuda_runtime.h>
#include <driver_types.h>

#define INF __int_as_float(0xff800000) 

/* 
Launch Configuration:
blockDim(BSZ, 1, 1)
gridDim(1, CEIL_DIV(N,BSZ), 1)

Assumes scores: N x d is stored in row-major format
        attn_scores: N x d stored in row-major format.
*/
template <const int BSZ>
__global__ void softmax(const float* scores , float* attn_scores, int N, int d){

    const int thread_row = threadIdx.x;
    const int global_row = blockIdx.y * BSZ + thread_row;

    if(global_row < N){
        float max_row_entry = -INF;
        for(int j = 0; j < d; ++j){
            max_row_entry = fmaxf(max_row_entry,scores[global_row * d + j]);
        }

        float row_sum = 0.0f;
        for(int j = 0; j < d; ++j){
            attn_scores[global_row * d + j] = expf(scores[global_row * d + j] - max_row_entry);
            row_sum += attn_scores[global_row * d + j];
        }

        for(int j = 0; j < d; ++j){
            attn_scores[global_row * d + j] /= row_sum;
        }
    }


}