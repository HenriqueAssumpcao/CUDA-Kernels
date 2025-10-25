#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>

#include <driver_types.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "device_launch_parameters.h"

// matrix

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
    private:
        size_t nrows,ncols,size;
        unsigned char device; // 0 is host, > 0 is CUDA

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
    
    public:
        float *data;
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
        void to(unsigned char device){
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

        matrix(const size_t nrows, const size_t ncols, unsigned char device = 0)
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
                
        // read-only get
        __host__ __device__ size_t get_nrows() const{
            return this->nrows;
        }
        __host__ __device__ size_t get_ncols() const{
            return this->ncols;
        }
        __host__ __device__ size_t get_size() const{
            return this->size;
        }
        __host__ __device__ bool get_device() const{
            return this->device;
        }

        // access matrix element

        __host__ __device__ float& operator()(const size_t row,const size_t col) {
            return this->data[row * this->ncols + col];
        }
        __host__ __device__ const float& operator()(const size_t row,const size_t col) const {
            return this->data[row * this->ncols + col];
        }
};

void print_matrix(const matrix &mat){
    if(!mat.get_device()){
        std::cout << "[";
        for(size_t i = 0; i < mat.get_nrows(); i++){
            std::cout << "[";
            for(size_t j = 0; j < mat.get_ncols()-1; j++){
                std::cout << (float)mat(i,j) << ",";
            }
            std::cout << (float)mat(i,mat.get_ncols()-1) << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
    else{
        std::cout << "Cant print matrix on device memory." << std::endl;
    }
}

// cpu
void rand_matrix(matrix &mat, const size_t n = 10, const float d = 1, const int seed=0){
    for(size_t i = 0; i < mat.get_nrows(); i++){
        for(size_t j = 0; j < mat.get_ncols(); j++){
            mat(i,j) = static_cast<float>(rand() % n)/d;
        }
    }
}
void matmul_naive(const matrix &x, const matrix &y, matrix &z){
    for(size_t i = 0; i < x.get_nrows(); i++){
        for(size_t j = 0; j < y.get_ncols(); j++){
            double dot = 0.0;
            for(size_t k = 0; k < x.get_ncols(); k++){
                dot += x(i,k)*y(k,j);
            }
            z(i,j) = dot;
        }
    }
}

// gpu

#define CEIL_DIV(x, y) (x + y - 1) / y

#define WARP_SIZE 32
#define HOST_ID 0
#define CUDA_ID 1

/*
Naive matrix multiplication kernel.
*/
__global__ void matmul_naive_kernel(const matrix x,const matrix y, matrix z){
    size_t col = blockDim.x*blockIdx.x + threadIdx.x;
    size_t row =blockDim.y*blockIdx.y + threadIdx.y;

    if(row < z.get_nrows() && col < z.get_ncols()){
        double dot = 0.0;
        for(size_t k = 0; k < x.get_ncols();k++){
            dot += x(row,k)*y(k,col);
        }
        z(row,col) = dot;
    }
}
/*
Matrix multiplication kernel with global memory coalescing.
*/
__global__ void matmul_coalesce_kernel(const matrix x, const matrix y, matrix z){
    size_t col = blockDim.x*blockIdx.x + (threadIdx.x % blockDim.x);
    size_t row = blockDim.x*blockIdx.y + (threadIdx.x / blockDim.x);

    if(row < z.get_nrows() && col < z.get_ncols()){
        double dot = 0.0;
        for(size_t k = 0; k < x.get_ncols();k++){
            dot += x(row,k)*y(k,col);
        }
        z(row,col) = dot;
    }
}
/*
Matrix multiplication kernel with shared memory tiling.
*/
template <const size_t BLOCK_SZ>
__global__ void matmul_sharedmem_kernel(const matrix x, const matrix y, matrix z){
    __shared__ float shared_x[BLOCK_SZ*BLOCK_SZ];
    __shared__ float shared_y[BLOCK_SZ*BLOCK_SZ];

    const uint tcol = threadIdx.x % BLOCK_SZ; // column of thread in tile
    const uint trow = threadIdx.x / BLOCK_SZ; // row of thread in tile

    const size_t zcol = BLOCK_SZ*blockIdx.x + tcol; // col in z we wish to compute
    const size_t zrow = BLOCK_SZ*blockIdx.y + trow; // row in z we wish to compute

    double dot = 0.0;

    for(size_t block_idx = 0; block_idx < CEIL_DIV(x.get_ncols(),BLOCK_SZ); ++block_idx){
        // load tile of A and B into shared memory
        if(zrow < x.get_nrows() && (block_idx*BLOCK_SZ + tcol) < x.get_ncols()){
            shared_x[trow*BLOCK_SZ + tcol] = x(zrow,block_idx*BLOCK_SZ + tcol);
        }
        else{
            shared_x[trow*BLOCK_SZ + tcol] = 0.0f;
        }
        if((block_idx*BLOCK_SZ + trow) < y.get_nrows() && zcol < y.get_ncols()){
            shared_y[trow*BLOCK_SZ + tcol] = y(block_idx*BLOCK_SZ + trow,zcol);
        }
        else{
            shared_y[trow*BLOCK_SZ + tcol] = 0.0f;
        }

        __syncthreads();

        for(size_t k = 0;k < BLOCK_SZ; k++){
            dot += shared_x[trow*BLOCK_SZ + k]*shared_y[k*BLOCK_SZ + tcol];
        }

        __syncthreads();
    }

    if(zrow < z.get_nrows() && zcol < z.get_ncols()){
        z(zrow,zcol) = dot;
    }
}

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

/*
Runs naive matrix multiplication kernel. Assumes inputs are on device.
*/
__host__ void matmul_naive_run(matrix &x, matrix &y, matrix &z){
    dim3 gridDim(CEIL_DIV(x.get_nrows(),WARP_SIZE),CEIL_DIV(y.get_ncols(),WARP_SIZE),1);
    dim3 blockDim(WARP_SIZE,WARP_SIZE,1);

    matmul_naive_kernel<<<gridDim,blockDim>>>(x,y,z);
}
/*
Runs matrix multiplication kernel with global memory coalescing. Assumes inputs are on device.
*/
__host__ void matmul_coalesce_run(matrix &x, matrix &y, matrix &z){
    dim3 gridDim(CEIL_DIV(x.get_nrows(),WARP_SIZE),CEIL_DIV(y.get_ncols(),WARP_SIZE),1);
    dim3 blockDim(WARP_SIZE*WARP_SIZE,1,1);

    matmul_coalesce_kernel<<<gridDim,blockDim>>>(x,y,z);
}

__host__ void matmul_sharedmem_run(matrix &x, matrix &y, matrix &z){
    dim3 gridDim(CEIL_DIV(x.get_nrows(),WARP_SIZE),CEIL_DIV(y.get_ncols(),WARP_SIZE),1);
    dim3 blockDim(WARP_SIZE*WARP_SIZE,1,1);

    matmul_sharedmem_kernel<WARP_SIZE><<<gridDim,blockDim>>>(x,y,z);   
}

__host__ void matmul_sharedmem_run2(matrix &x, matrix &y, matrix &z){
    dim3 gridDim(CEIL_DIV(x.get_nrows(),WARP_SIZE),CEIL_DIV(y.get_ncols(),WARP_SIZE),1);
    dim3 blockDim(WARP_SIZE*WARP_SIZE,1,1);

    sgemm_shared_mem_block<WARP_SIZE><<<gridDim,blockDim>>>(x.get_nrows(),y.get_ncols(),x.get_ncols(),1.0,x.data,y.data,0.0,z.data);   
}



// benchmark

void benchmark_matmul_cpu(const size_t nruns,const size_t n,const size_t m, const size_t p, const int seed){
    std::cout << "[CPU] MATMUL" << std::endl;
    matrix x(n,m),y(m,p),xpy(n,p); // all on host by default

    std::vector<float> durations(nruns);
    for(size_t crun = 0; crun < nruns; crun++){
        srand(seed+crun);
        rand_matrix(x);rand_matrix(y);

        auto start = std::chrono::high_resolution_clock::now();

        matmul_naive(x,y,xpy);

        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = stop - start;

        durations[crun] = duration.count() * 1000.0f;
    }
    double total_rt = 0.0;
    for(size_t i = 0; i < nruns; i++){
        total_rt += durations[i];
    }
    std::cout << "Total matmul time: " << total_rt << " ms" << std::endl;
    std::cout << "Average matmul time: " << total_rt/(double)nruns << " ms" << std::endl;

    x.memfree();y.memfree();xpy.memfree();
}

void benchmark_matmul_kernel(const size_t nruns,const size_t n,const size_t m, const size_t p, const int seed,
                             void(*run)(matrix&, matrix&, matrix&)){
    std::cout << "[GPU] MATMUL" << std::endl;
    matrix x(n,m),y(m,p),xpy(n,p); // all on host by default

    std::vector<float> durations(nruns);
    for(size_t crun = 0; crun < nruns; crun++){
        srand(seed+crun);
        rand_matrix(x);rand_matrix(y);

        auto start = std::chrono::high_resolution_clock::now();
        
        x.to(CUDA_ID);y.to(CUDA_ID);xpy.to(CUDA_ID);
        run(x,y,xpy);
        x.to(HOST_ID);y.to(HOST_ID);xpy.to(HOST_ID);

        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = stop - start;

        durations[crun] = duration.count() * 1000.0f;
    }
    double total_rt = 0.0;
    for(size_t i = 0; i < nruns; i++){
        total_rt += durations[i];
    }
    std::cout << "Total matmul time: " << total_rt << " ms" << std::endl;
    std::cout << "Average matmul time: " << total_rt/(double)nruns << " ms" << std::endl;

    x.memfree();y.memfree();xpy.memfree();
}


int main(int argc, char* argv[]){
    if(argc < 6){ 
        std::cout << "Usage: " << argv[0] << " <num_runs> <n> <m> <p> <seed>" << std::endl;
        return 1;
    }
    
    try {
        size_t nruns = std::stoull(argv[1]);
        size_t n = std::stoull(argv[2]);
        size_t m = std::stoull(argv[3]); 
        size_t p = std::stoull(argv[4]);
        int seed = std::stoi(argv[5]);

        benchmark_matmul_kernel(nruns,n,m,p,seed,matmul_naive_run);
        benchmark_matmul_kernel(nruns,n,m,p,seed,matmul_sharedmem_run);
        // benchmark_matmul_kernel(nruns,n,m,p,seed,matmul_sharedmem_run2);
        benchmark_matmul_kernel(nruns,n,m,p,seed,matmul_coalesce_run);
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}