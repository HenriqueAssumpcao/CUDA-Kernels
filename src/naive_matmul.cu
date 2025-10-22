#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <numeric>

#include <driver_types.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_runtime_api.h"
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

#define MAJOR_AXIS 1 // > 1 row major, 0 column major

struct matrix{
    private:
        size_t nrows,ncols,size;
        bool device;
        float *arr;
        void memfree(){
            if(this->arr){
                if(this->device){
                    CUDA_CHECK_ERROR(cudaFree(this->arr));
                }
                else{
                    free(this->arr);
                }
                this->arr = nullptr;
                this->device = 0;
            }
        }

        void alloc_host(){
            if(this->arr){
                this->memfree();
            }
            this->device = 0;
            this->arr = (float*)malloc(this->size);
        }

        void alloc_device(){
            if(this->arr){
                this->memfree();
            }
            this->device = 1;
            CUDA_CHECK_ERROR(cudaMalloc(&this->arr, this->size));
        }

    public:
        matrix(){
            this->nrows = 0;this->ncols = 0;this->size=0;
            this->device = 0;
            this->arr = nullptr;
        }
        matrix(size_t nrows, size_t ncols, bool device = 0){
            this->nrows = nrows;this->ncols = ncols;
            this->size = nrows*ncols*sizeof(float);
            this->arr = nullptr;

            if(device){ // device
                this->alloc_device();
            }
            else{ // host
                this->alloc_host();
            }
        }

        ~matrix(){
            if(this->arr){
                this->memfree();
            }
        }

        size_t get_nrows(){
            return this->nrows;
        }
        size_t get_ncols(){
            return this->ncols;
        }
        size_t get_size(){
            return this->size;
        }
        bool get_device(){
            return this->device;
        }

        size_t get_idx(const size_t row, const size_t col){
            if(MAJOR_AXIS){
                return row*this->ncols + col;
            }
            else{
                return col*this->nrows + row;
            }
        }

        float get(const size_t row, const size_t col){
            return this->arr[this->get_idx(row,col)];
        }

        void set(const size_t row, const size_t col, const float val){
            size_t idx = this->get_idx(row,col);
            if(idx < this->size){
                this->arr[idx] = val;
            }
        }

        void to_host(){
            if(this->arr && this->device){
                float *h_arr = (float*)malloc(this->size);
                CUDA_CHECK_ERROR(cudaMemcpy(h_arr, this->arr, this->size, cudaMemcpyDeviceToHost));
                this->memfree();
                this->arr = h_arr;
            }
        }

        void to_device(){
            if(this->arr && !this->device){
                float *d_arr;
                CUDA_CHECK_ERROR(cudaMalloc(&d_arr, this->size));
                CUDA_CHECK_ERROR(cudaMemcpy(d_arr, this->arr, this->size, cudaMemcpyHostToDevice));
                this->memfree();
                this->arr = d_arr;
            }
        }

        
};

// utils
void print_matrix(matrix &mat){
    if(!mat.get_device()){
        std::cout << "[";
        for(size_t i = 0; i < mat.get_nrows(); i++){
            std::cout << "[";
            for(size_t j = 0; j < mat.get_ncols()-1; j++){
                std::cout << (float)mat.get(i,j) << ",";
            }
            std::cout << (float)mat.get(i,mat.get_ncols()-1) << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
    else{
        std::cout << "Cannot cout matrix on device" << std::endl;
    }
}

void rand_matrix(matrix &mat, const size_t n = 10, const float d = 1, const int seed=0){
    for(size_t i = 0; i < mat.get_nrows(); i++){
        for(size_t j = 0; j < mat.get_ncols(); j++){
            mat.set(i,j,static_cast<float>(rand() % n)/d);
        }
    }
}

// CPU operations
void transpose(matrix &mat, matrix &mat_t){
    for(size_t i = 0; i < mat.get_nrows(); i++){
        for(size_t j = 0; j < mat.get_ncols(); j++){
            mat_t.set(j,i,mat.get(i,j));
        }
    }
}

void matmul(matrix &x, matrix &y, matrix &z){
    for(size_t i = 0; i < x.get_nrows(); i++){
        for(size_t j = 0; j < y.get_ncols(); j++){
            double dot = 0;
            for(size_t k = 0 ; k < x.get_ncols(); k++){
                dot += x.get(i,k)*y.get(k,j);
            }
            z.set(i,j,(float)dot);
        }
    }
}


// benchmark CPU
std::vector<double> benchmark_matmul(const size_t nruns,const size_t n,const size_t m, const int seed){
    std::vector<double> durations(nruns);
    for(size_t crun = 0; crun < nruns; crun++){
        srand(seed+crun);

        matrix x(n,m),y(m,n),xpy(n,n); // all on host by default

        rand_matrix(x);rand_matrix(y);

        auto start = std::chrono::high_resolution_clock::now();
        matmul(x,y,xpy);

        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = stop - start;

        durations[crun] = duration.count() * 1000.0f;
    }
    double total_rt = std::accumulate(durations.begin(),durations.end(),0);
    std::cout << "Total matmul time: " << total_rt << " ms" << std::endl;
    std::cout << "Average matmul time: " << total_rt/(double)nruns << " ms" << std::endl;

    return durations;
}

// naive GPU matmul

__global__ void matmul_naive_kernel(matrix x, matrix y, matrix z){
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    double dot = 0;
    if(i < x.get_nrows() && j < y.get_ncols()){
        for(size_t k = 0; k < x.get_ncols(); k++){
            dot += x.get(i,k)*y.get(k,j);
        }
        z.set(i,j,dot);
    }
}

int main(int argc, char* argv[]){
    if(argc < 5){
        std::cout << "Expected (num_runs,n,m,seed) as argv" << std::endl;
        return 1;
    }

    size_t nruns = (size_t)std::stoi(argv[1]), n = (size_t)std::stoi(argv[2]),m = (size_t)std::stoi(argv[3]);
    int seed = std::stoi(argv[4]);

    srand(seed);
    matrix x(n,m,0),y(m,n,0),xpy(n,n,0); // all on host by default

    rand_matrix(x);rand_matrix(y);

    print_matrix(x);

    std::vector<double> durations = benchmark_matmul(nruns,n,m,seed);

    return 0;
}