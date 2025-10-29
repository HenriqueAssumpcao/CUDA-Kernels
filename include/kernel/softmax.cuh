#pragma once

#include <stdlib.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <driver_types.h>

#define INF __int_as_float(0xff800000) 

template <const uint BLOCK_SZ>
__global__ void softmax(const float* attn_scores , float* softmax_scores, int N){

    const uint thread_row = threadIdx.x;
    const uint global_row = blockIdx.y * BLOCK_SZ + thread_row;

    if(global_row < M){
        float max_row_entry = -INF;
        for(uint j = 0; j < N; ++j){
            max_row_entry = fmaxf(max_row_entry,attn_scores[global_row * N + j]);
        }

        float row_sum = 0.0f;
        for(uint j = 0; j < N; ++j){
            softmax_scores[global_row * N + j] = expf(attn_scores[global_row * N + j] - max_row_entry);
            row_sum += softmax_scores[global_row * N + j];
        }

        for(uint j = 0; j < N; ++j){
            softmax_scores[global_row * N + j] /= row_sum;
        }
    }


}