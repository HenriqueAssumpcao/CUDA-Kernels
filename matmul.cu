#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

// utils

void print_dmat(double *mat,const size_t nrows, const size_t ncols){
    cout << "[";
    for(int i = 0; i < nrows; i++){
        cout << "[";
        for(int j = 0; j < ncols-1; j++){
            cout << mat[i*nrows + j] << ",";
        }
        cout << mat[i*nrows + (ncols-1)] << "]" << endl;
    }
    cout << "]" << endl;
}

void rand_dmat(double *mat,const size_t nrows, const size_t ncols){
    size_t i,j;
    for(i = 0; i < nrows; i++){
        for(j = 0; j < ncols; j++){
            mat[i*nrows + j] = static_cast<double>(rand() % 1000)/1000.0;
        }
    }
}

// CPU naive matmul

__host__ void h_matmul(const double *A, const double *B, double *C, const size_t n, const size_t m, const size_t t){
    size_t i,j,k;
    for(i = 0; i < n; i++){
        for(j = 0; j < t; j++){
            double dot = 0;
            for(k = 0; k < m; k++){
                dot += A[i*n + k]*B[k*m + j];
            }
            C[i*n + j] = dot;
        }
    }
}

// GPU naive matmul 

__global__ void d_matmul(const double *A, const double *B, double *C, const size_t n, const size_t m, const size_t t){
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < n){
        size_t j,k;
        for(j = 0; j < t; j++){
            double dot = 0;
            for(k = 0; k < m; k++){
                dot += A[idx*n + k]*B[k*m + j];
            }
            C[idx*n + j] = dot;
        }
    }
}

int main(int argc, char* argv[]){
    int seed = 2025;
    srand(seed);

    if(argc < 3){
        cout << "Expected n m as argv" << endl;
        return 1;
    }

    size_t n = (size_t)stoi(argv[1]),m = (size_t)stoi(argv[2]);

    double *a = (double*)malloc((n*m)*sizeof(double));
    double *b = (double*)malloc((m*n)*sizeof(double));

    rand_dmat(a,n,m);rand_dmat(b,m,n);

    print_dmat(a,n,m);
    print_dmat(b,m,n);

    double *c = (double*)malloc((n*n)*sizeof(double));

    h_matmul(a,b,c,n,m,n);

    print_dmat(c,n,n);




    free(a);free(b);free(c);

    return 0;
}

