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
#define CEIL_DIV(x, y) (x + y - 1) / y

// KERNELS
/*
Naive matmul kernel.
Assumes row-major arrays.
A: M x K
B: K x N
C: M x N
*/
__global__ void matmul_naive(const float *A,const float *B,float *C,
                             int M, int K, int N)
{
    size_t trow = threadIdx.y;
    size_t tcol = threadIdx.x;

    size_t row =  blockDim.y*blockIdx.y + trow;
    size_t col = blockDim.x*blockIdx.x + tcol;

    if(row < M && col < N){
        double dot = 0.0;
        for(size_t k = 0; k < K; k++){
            dot += A[row*K + k]*B[k*N + col];
        }
        C[row*N + col] = dot;
    }
}

/*
Global Memory Coalescing matmul kernel.
*/
template <const int WARP_SZ>
__global__ void matmul_coalesce(const float *A,const float *B,float *C,
                                int M, int K, int N)
{
    size_t trow = threadIdx.x / WARP_SZ;
    size_t tcol = threadIdx.x % WARP_SZ;

    size_t row = blockDim.y*blockIdx.y + trow;
    size_t col = blockDim.x*blockIdx.x + tcol;

    if(row < M && col < N){
        double dot = 0.0;
        for(size_t k = 0; k < K; k++){
            dot += A[row*K + k]*B[k*N + col];
        }
        C[row*N + col] = dot;
    }
}

/*
Shared memory and gmc matmul kernel.
Assumes blockDim(WARP_SZ*WARP_SZ,1,1),gridDim(CEIL_DIV(M,WARP_SZ),CEIL_DIV(N,WARP_SZ),1).
*/
template <const int WARP_SZ>
__global__ void matmul_shared(const float *A,const float *B,float *C,
                                int M, int K, int N)
{
    size_t trow = threadIdx.x / WARP_SZ;
    size_t tcol = threadIdx.x % WARP_SZ;

    __shared__ float s_A[WARP_SZ*WARP_SZ],s_B[WARP_SZ*WARP_SZ];

    A += blockIdx.y * WARP_SZ * K;
    B += blockIdx.x * WARP_SZ;
    C += (blockIdx.y * WARP_SZ * N) + (blockIdx.x * WARP_SZ);
    
    double dot = 0.0;

    for(size_t bIdx = 0; bIdx < K; bIdx += WARP_SZ){
        s_A[trow*WARP_SZ + tcol] = A[trow*K + tcol];
        s_B[trow*WARP_SZ + tcol] = B[trow*N + tcol];

        __syncthreads();

        for(size_t k = 0;k < WARP_SZ; ++k){
            dot += s_A[trow*WARP_SZ + k]*s_B[k*WARP_SZ + tcol];
        }

        A += WARP_SZ;
        B += (WARP_SZ*N);

        __syncthreads();   

    }
    C[trow*N + tcol] = dot;
}



// template <const int WARP_SZ>
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