#pragma once

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

#define CUDA_ID 1
#define HOST_ID 0

template <typename RunKernelFunc>
void benchmark_kernel(const size_t nruns, const size_t M, const size_t N,
                      const size_t K, const int seed, const float TOL,
                      RunKernelFunc run_kernel,
                      const std::string &kernel_name = "") {
    // Create CUDA events for accurate GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Pre-allocate matrices once
    matrix A(M, K), B(K, N), C(M, N);
    srand(seed);
    rand_matrix(A);
    rand_matrix(B);
    A.to(CUDA_ID);
    B.to(CUDA_ID);
    C.to(CUDA_ID);

    // warmup
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

    std::cout << kernel_name << " Runtime (ms): " << mean << " ± " << std_dev
              << " [min: " << runtimes[0] << ", max: " << runtimes.back() << "]"
              << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    A.memfree();
    B.memfree();
    C.memfree();
}