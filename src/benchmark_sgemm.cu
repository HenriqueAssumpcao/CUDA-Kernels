#include <algorithm>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <cuda_runtime.h>
#include <driver_types.h>

#include "matrix.cuh"
#include "utils.cuh"
#include "runners.cuh"

template <typename RunKernelFunc>
void benchmark_sgemm_kernel(const size_t nruns, const size_t M, const size_t N,
                      const size_t K, const int seed, const float TOL,
                      RunKernelFunc run_kernel,
                      const std::string &kernel_name = "") {

    std::cout << "\n============================================================" << std::endl;
    std::cout << "Benchmarking Kernel: " << kernel_name << std::endl;
    std::cout << "Matrix Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    matrix A(M, K), B(K, N), C(M, N);
    srand(seed);
    rand_matrix(A);
    rand_matrix(B);
    A.to(CUDA_ID);
    B.to(CUDA_ID);
    C.to(CUDA_ID);

    const size_t warmup_runs = std::min(nruns / 10, size_t(10));
    for (size_t i = 0; i < warmup_runs; ++i) {
        run_kernel(A.data, B.data, C.data, M, K, N, 1, 0);
        cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();

    std::vector<float> runtimes(nruns);

    for (size_t runIdx = 0; runIdx < nruns; ++runIdx) {
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        run_kernel(A.data, B.data, C.data, M, K, N, 1, 0);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        runtimes[runIdx] = milliseconds;

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error in " << kernel_name << ": "
                      << cudaGetErrorString(err) << std::endl;
            break;
        }
    }

    std::sort(runtimes.begin(), runtimes.end());

    float mean, std_dev;
    compute_stats(runtimes.data(), runtimes.size(), mean, std_dev);
    std::cout << "Runtime (ms): " << mean << " ± " << std_dev
              << " [min: " << runtimes[0] << ", max: " << runtimes.back() << "]"
              << std::endl;
    std::cout << "============================================================" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    A.memfree();
    B.memfree();
    C.memfree();
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        std::cout << "Usage: " << argv[0] << " <num_runs> <M> <N> <K> <seed>"
                  << std::endl;
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
            std::make_pair(run_sgemm_naive<32>, "Naive sgemm (32)"),
            std::make_pair(run_sgemm_blocktiling<32, 32, 32>,
                           "Block tiling sgemm (32,32,32)"),
            std::make_pair(run_sgemm_blocktiling<64, 8, 64, 4, 4>,
                        "Thread tiling sgemm (64,8,64,4,4)")
                           };

        for (auto &[kernel_func, name] : kernels) {
            benchmark_sgemm_kernel(nruns, M, N, K, seed, TOL, kernel_func, name);
            cudaDeviceSynchronize();
        }

    } catch (const std::exception &e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}