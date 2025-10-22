#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// matrix of element of type T
struct matrix{
    size_t nrows,ncols;
    size_t arr_size;
    bool major_axis; // 0 for row major, 1 for column major
    double *h_arr,*d_arr;

    matrix(const size_t nrows, const size_t ncols, bool major_axis=0){
        this->nrows = nrows;this->ncols = ncols;
        this->major_axis = major_axis;

        this->arr_size = nrows*ncols*sizeof(double);

        this->h_arr = (double*)malloc(this->arr_size);
        CUDA_CHECK_ERROR(cudaMalloc(&this->d_arr,this->arr_size));
    }

    size_t map(const size_t r, const size_t c){
        if(major_axis == 0){
            return (r*this->ncols) + c;
        }
        else{
            return (c*this->nrows) + r;
        }
    }

    ~matrix(){
        free(this->h_arr);
        this->h_arr = nullptr;

        CUDA_CHECK_ERROR(cudaFree(this->d_arr));
        this->d_arr = nullptr;
    }
};