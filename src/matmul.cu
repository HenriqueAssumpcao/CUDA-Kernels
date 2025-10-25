#include "matrix.h"

#define WARP_SIZE 32
#define CEIL_DIV(x, y) (x + y - 1) / y


__global__ void 


/*
Naive matrix multiplication kernel.
*/
__global__ void matmul_naive_kernel(const matrix x,const matrix y, matrix z){
    size_t col = blockDim.x*blockIdx.x + threadIdx.x,row =blockDim.y*blockIdx.y + threadIdx.y;

    if(row < z.get_nrows() && col < z.get_ncols()){
        double dot = 0.0;
        for(size_t k = 0; k < x.get_ncols();k++){
            dot += x(row,k)*y(k,col);
        }
        z(row,col) = dot;
    }
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
Runs matrix multiplication kernel with global memory coalescing. Assumes inputs are on device.
*/
__host__ void matmul_coalesce_run(matrix &x, matrix &y, matrix &z){
    dim3 gridDim(CEIL_DIV(x.get_nrows(),WARP_SIZE),CEIL_DIV(y.get_ncols(),WARP_SIZE),1);
    dim3 blockDim(WARP_SIZE*WARP_SIZE,1,1);

    matmul_coalesce_kernel<<<gridDim,blockDim>>>(x,y,z);
}