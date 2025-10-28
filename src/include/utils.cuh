#pragma once
 
#include <cmath>
#include <iostream>
#include <assert.h>

void sgemm_cpu(const float *A,
               const float *B,
               float *C,
               int M, int K, int N,
               float alpha = 1, float beta = 0){

    for(size_t i = 0; i < M; ++i){
        for(size_t j = 0; j < N; ++j){
            double dot = 0.0;
            for(size_t k = 0; k < K; ++k){
                dot += A[i*K + k]*B[k*N + j];
            }
            C[i*N + j] = alpha * dot + beta * C[i*N + j];
        }
    }
}

bool matrices_close(const matrix &A,const matrix &B, float tolerance) {
    assert(A.size == B.size);
    for (int i = 0; i < (A.nrows * A.ncols); ++i) {
        if (fabs(A.data[i] - B.data[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void compute_stats(const float data[], const int size, float& mean, float& std_dev) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum += data[i];
    }
    mean = sum / size;

    float variance = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    std_dev = sqrtf(variance / size);
}

void rand_matrix(matrix &mat, float min_val = 0.0f, float max_val = 1.0f) {
    float range = max_val - min_val;
    
    for (size_t i = 0; i < (mat.nrows*mat.ncols); i++) {
        mat.data[i] = min_val + (float)rand() / RAND_MAX * range;
    }
}

void zero_matrix(matrix &mat){
    for(size_t i = 0; i < mat.nrows; i++){
        for(size_t j = 0; j < mat.ncols; j++){
            mat.data[i*mat.ncols + j] = 0;
        }
    }
}

void print_matrix(const matrix &mat){
    if(!mat.device){
        std::cout << "[";
        for(size_t i = 0; i < mat.nrows; i++){
            std::cout << "[";
            for(size_t j = 0; j < mat.ncols-1; j++){
                std::cout << (float)mat.data[i*mat.ncols + j] << ",";
            }
            std::cout << (float)mat.data[i*mat.ncols + (mat.ncols - 1)] << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
    else{
        std::cout << "Cant print matrix on device memory." << std::endl;
    }
}