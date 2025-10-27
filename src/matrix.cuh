#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <driver_types.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
Struct representing matrix of size nrows x ncols, stored in row-major format.
*/
struct matrix{
        size_t nrows,ncols,size;
        bool device; // 0 is host, > 0 is CUDA
        float *data;

        /*
        Allocates this->size bytes of host memory.
        */
        void malloc_host(){
            if(this->data){
                this->memfree();
            }
            this->data = (float*)malloc(this->size);
            this->device = 0;
        }

        /*
        Allocates this->size bytes of device memory.
        */
        void malloc_device(){
            if(this->data){
                this->memfree();
            }
            CUDA_CHECK_ERROR(cudaMalloc(&this->data, this->size));
            this->device = 1;
        }

        /*
        Frees memory allocated by object.
        */
        void memfree(){
            if(this->data){
                if(this->device){
                    CUDA_CHECK_ERROR(cudaFree(this->data));
                }
                else{
                    free(this->data);
                }
                this->data = nullptr;
                this->device = 0;
            }
        }
        void to(bool device){
            if(!this->data || (device == this->device)){
                return;
            }
            if(device){
                float *d_arr;
                CUDA_CHECK_ERROR(cudaMalloc(&d_arr, this->size));
                CUDA_CHECK_ERROR(cudaMemcpy(d_arr, this->data, this->size, cudaMemcpyHostToDevice));
                this->memfree();
                this->data = d_arr;
                this->device = 1;
            }
            else{
                float *h_arr = (float*)malloc(this->size);
                CUDA_CHECK_ERROR(cudaMemcpy(h_arr, this->data, this->size, cudaMemcpyDeviceToHost));
                this->memfree();
                this->data = h_arr;
                this->device = 0;
            }
        }

        matrix()
            : nrows(0), ncols(0), size(0), 
            device(0), data(nullptr){}

        matrix(const size_t nrows, const size_t ncols, bool device = 0)
            : nrows(nrows), ncols(ncols), size(nrows * ncols * sizeof(float)), 
            device(0), data(nullptr) {
            if (device) {
                this->malloc_device();
            } else {
                this->malloc_host();
            }
        }

        matrix(const matrix& other) 
            : nrows(other.nrows), ncols(other.ncols), size(other.size), 
            device(false), data(nullptr) { 
            
            if (other.data) {
                if (other.device) {
                    this->malloc_device();
                    CUDA_CHECK_ERROR(cudaMemcpy(data, other.data, size, cudaMemcpyDeviceToDevice));
                } else {
                    this->malloc_host();
                    memcpy(data, other.data, size);
                }
            }
        }

        matrix& operator=(const matrix& other) {
            if (this != &other) {
                this->memfree();

                nrows = other.nrows;
                ncols = other.ncols;
                size = other.size;
                
                if (other.data) {
                    if (other.device) {
                        this->malloc_device();
                        CUDA_CHECK_ERROR(cudaMemcpy(data, other.data, size, cudaMemcpyDeviceToDevice));
                    } else {
                        this->malloc_host();
                        memcpy(data, other.data, size);
                    }
                }
            }
            return *this;
        }
};


// template <const int BLOCK_SZ>
// __host__ void run_matmul_kernel(const float *h_A,const float *h_B,float *h_C,
//                                 int M, int K, int N,
//                                 void(*matmul_kernel)(const float*, const float*, float*, int, int, int),
//                                 const dim3 gridDim, const dim3 blockDim)
// {
//     // alloc and copy to device
//     float *d_A,*d_B,*d_C;

//     CUDA_CHECK_ERROR(cudaMalloc(&d_A,sizeof(h_A)));
//     CUDA_CHECK_ERROR(cudaMemcpy(d_A,h_A,sizeof(h_A),cudaMemcpyHostToDevice));

//     CUDA_CHECK_ERROR(cudaMalloc(&d_B,sizeof(h_B)));
//     CUDA_CHECK_ERROR(cudaMemcpy(d_B,h_B,sizeof(h_B),cudaMemcpyHostToDevice));

//     CUDA_CHECK_ERROR(cudaMalloc(&d_C,sizeof(h_C)));
//     CUDA_CHECK_ERROR(cudaMemcpy(d_C,h_C,sizeof(h_C),cudaMemcpyHostToDevice));

//     // run kernel
//     matmul_kernel<<<gridDim,blockDim>>>(d_A,d_B,d_C,M,N,K);

//     // back to host
//     CUDA_CHECK_ERROR(cudaMemcpy(h_A,d_A,sizeof(d_A),cudaMemcpyDeviceToHost));
//     CUDA_CHECK_ERROR(cudaMemcpy(h_B,d_B,sizeof(d_B),cudaMemcpyDeviceToHost));
//     CUDA_CHECK_ERROR(cudaMemcpy(h_C,d_C,sizeof(d_C),cudaMemcpyDeviceToHost));

//     // free on device
//     CUDA_CHECK_ERROR(cudaFree(d_A));
//     CUDA_CHECK_ERROR(cudaFree(d_B));
//     CUDA_CHECK_ERROR(cudaFree(d_C));
// }