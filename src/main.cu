#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <driver_types.h>
#include <cuda_runtime.h>

#include "include/benchmark.cuh"
#include "include/runners.cuh"

#define CUDA_ID 1
#define HOST_ID 0


int main(int argc, char* argv[]){
    if(argc < 6){ 
        std::cout << "Usage: " << argv[0] << " <num_runs> <M> <N> <K> <seed>" << std::endl;
        return 1;
    }
    
    try {
        const size_t nruns = std::stoul(argv[1]);
        const size_t M = std::stoul(argv[2]);
        const size_t N = std::stoul(argv[3]); 
        const size_t K = std::stoul(argv[4]);
        const int seed = std::stoi(argv[5]);

        const float TOL = 1e-5f;

        auto kernels = {
            std::make_pair(run_matmul_naive<32>, "Naive matmul"),
            std::make_pair(run_matmul_naive_ref<32>, "Naive matmul ref"),
            std::make_pair(run_matmul_coalesce<32>, "Coalesce matmul"),
            std::make_pair(run_matmul_coalesce_ref<32>, "Coalesce matmul ref"),
            std::make_pair(run_matmul_shared<32>, "Shared matmul"),
            std::make_pair(run_matmul_shared_ref<32>, "Shared matmul ref")
        };


        for(auto& [kernel_func, name] : kernels) {
            benchmark_kernel(nruns, M, N, K, seed, TOL, kernel_func, name);
            cudaDeviceSynchronize();
        }   
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}