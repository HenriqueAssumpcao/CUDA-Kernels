#pragma once

#include <stdlib.h>

#include <assert.h>
#include <cmath>
#include <iostream>

#include <cuda_runtime.h>
#include <driver_types.h>

#define CEIL_DIV(x, y) (x + y - 1) / y

/**
 * @brief Macro for automatic CUDA error checking
 * @param ans CUDA function call to check for errors
 *
 * Wraps CUDA API calls and automatically checks for errors using gpuAssert.
 * Usage: CUDA_CHECK_ERROR(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK_ERROR(ans)                                                  \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }
/**
 * @brief Internal function for CUDA error assertion and reporting
 * @param code CUDA error code returned from CUDA API call s
 * @param file Source file where the error occurred
 * @param line Line number where the error occurred
 * @param abort Whether to terminate program on error (default: true)
 *
 * Prints detailed error information and optionally terminates the program
 * if a CUDA error is detected.
 */
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

#define CUDA_ID 1
#define HOST_ID 0