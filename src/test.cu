#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

void print_vec(const int *x, const size_t size){
    size_t i;
    printf("vec(");

    for(i = 0; i < size-1; i++){
        printf("%d,",x[i]);
    }

    printf("%d)\n",x[size-1]);
}

__global__ void add_ivec(const int *x, const int *y, int *res, const size_t size){
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < size){
        res[idx] = x[idx] + y[idx];
    }
}


int main(){
    //
    int seed = 2025;
    srand(seed);


    // host
    const size_t int_bytes = sizeof(int);
    const size_t vec_size = 32;

    int *h_x = (int*)malloc(vec_size * int_bytes);
    int *h_y = (int*)malloc(vec_size * int_bytes);
    int *h_res = (int*)malloc(vec_size * int_bytes);

    size_t i;
    for(i = 0; i < vec_size; i++){
        h_x[i] = rand() % 100;
        h_y[i] = rand() % 100;
        h_res[i] = 0;
    }
    print_vec(h_x,vec_size);
    print_vec(h_y,vec_size);

    // device

    int *d_x,*d_y,*d_res;

    cudaError_t err; 
    
    err = cudaMalloc(&d_x,int_bytes*vec_size);
    err = cudaMalloc(&d_y,int_bytes*vec_size);
    err = cudaMalloc(&d_res,int_bytes*vec_size);

    cudaMemcpy(d_x, h_x, vec_size * int_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, vec_size * int_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res, vec_size * int_bytes, cudaMemcpyHostToDevice);

    dim3 block_shape(16);
    dim3 grid_shape((block_shape.x + vec_size - 1)/block_shape.x);

    add_ivec<<<block_shape,grid_shape>>>(d_x,d_y,d_res,vec_size);

    cudaDeviceSynchronize();

    cudaMemcpy(h_x,d_x,vec_size * int_bytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y,d_y,vec_size * int_bytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res,d_res,vec_size * int_bytes,cudaMemcpyDeviceToHost);

    print_vec(h_res,vec_size);

    // free

    free(h_x);
    free(h_y);
    free(h_res);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);

    return 0;
}