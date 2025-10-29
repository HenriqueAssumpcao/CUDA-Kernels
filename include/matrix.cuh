#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <driver_types.h>

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

/**
 * @brief Matrix structure supporting both host and device memory management
 *
 * Provides a unified interface for 2D float matrices that can reside in either
 * host (CPU) or device (GPU) memory with automatic memory management and
 * seamless transfers between memory spaces.
 */
struct matrix {
    size_t nrows, ncols,
        size;    ///< Matrix dimensions and total memory size in bytes
    bool device; ///< Memory location flag (0=host, 1=device)
    float *data; ///< Pointer to matrix data

    /**
     * @brief Allocate matrix memory on the host (CPU)
     *
     * Frees any existing memory allocation before allocating new host memory.
     * Sets device flag to false.
     */
    void malloc_host() {
        if (this->data) {
            this->memfree();
        }
        this->data = (float *)malloc(this->size);
        this->device = 0;
    }

    /**
     * @brief Allocate matrix memory on the device (GPU)
     *
     * Frees any existing memory allocation before allocating new device memory.
     * Sets device flag to true.
     */
    void malloc_device() {
        if (this->data) {
            this->memfree();
        }
        CUDA_CHECK_ERROR(cudaMalloc(&this->data, this->size));
        this->device = 1;
    }

    /**
     * @brief Free allocated matrix memory
     *
     * Automatically detects whether memory is on host or device and calls
     * the appropriate deallocation function. Resets data pointer to nullptr
     * and device flag to false.
     */
    void memfree() {
        if (this->data) {
            if (this->device) {
                CUDA_CHECK_ERROR(cudaFree(this->data));
            } else {
                free(this->data);
            }
            this->data = nullptr;
            this->device = 0;
        }
    }

    /**
     * @brief Transfer matrix between host and device memory
     * @param device Target memory location (true=device, false=host)
     *
     * Transfers matrix data between host and device memory. If the matrix
     * is already in the target location or data is null, the function returns
     * without performing any operation.
     */
    void to(bool device) {
        if (!this->data || (device == this->device)) {
            return;
        }
        if (device) {
            float *d_arr;
            CUDA_CHECK_ERROR(cudaMalloc(&d_arr, this->size));
            CUDA_CHECK_ERROR(cudaMemcpy(d_arr, this->data, this->size,
                                        cudaMemcpyHostToDevice));
            this->memfree();
            this->data = d_arr;
            this->device = 1;
        } else {
            float *h_arr = (float *)malloc(this->size);
            CUDA_CHECK_ERROR(cudaMemcpy(h_arr, this->data, this->size,
                                        cudaMemcpyDeviceToHost));
            this->memfree();
            this->data = h_arr;
            this->device = 0;
        }
    }

    /**
     * @brief Default constructor
     *
     * Creates an empty matrix with zero dimensions and null data pointer.
     */
    matrix() : nrows(0), ncols(0), size(0), device(0), data(nullptr) {}

    /**
     * @brief Parameterized constructor
     * @param nrows Number of matrix rows
     * @param ncols Number of matrix columns
     * @param device Memory location (default: false for host memory)
     *
     * Creates a matrix with specified dimensions and allocates memory
     * on either host or device based on the device parameter.
     */
    matrix(const size_t nrows, const size_t ncols, bool device = 0)
        : nrows(nrows), ncols(ncols), size(nrows * ncols * sizeof(float)),
          device(0), data(nullptr) {
        if (device) {
            this->malloc_device();
        } else {
            this->malloc_host();
        }
    }

    /**
     * @brief Copy constructor
     * @param other Matrix to copy from
     *
     * Creates a deep copy of another matrix, preserving the original's
     * memory location (host or device) and copying all data.
     */
    matrix(const matrix &other)
        : nrows(other.nrows), ncols(other.ncols), size(other.size),
          device(false), data(nullptr) {

        if (other.data) {
            if (other.device) {
                this->malloc_device();
                CUDA_CHECK_ERROR(cudaMemcpy(data, other.data, size,
                                            cudaMemcpyDeviceToDevice));
            } else {
                this->malloc_host();
                memcpy(data, other.data, size);
            }
        }
    }

    /**
     * @brief Assignment operator
     * @param other Matrix to assign from
     * @return Reference to this matrix
     *
     * Performs deep copy assignment, freeing existing memory and copying
     * dimensions, memory location, and data from the source matrix.
     * Handles self-assignment safely.
     */
    matrix &operator=(const matrix &other) {
        if (this != &other) {
            this->memfree();

            nrows = other.nrows;
            ncols = other.ncols;
            size = other.size;

            if (other.data) {
                if (other.device) {
                    this->malloc_device();
                    CUDA_CHECK_ERROR(cudaMemcpy(data, other.data, size,
                                                cudaMemcpyDeviceToDevice));
                } else {
                    this->malloc_host();
                    memcpy(data, other.data, size);
                }
            }
        }
        return *this;
    }

    /**
     * @brief Destructor
     *
     * Automatically frees allocated memory when the matrix object
     * goes out of scope, preventing memory leaks.
     */
    ~matrix() { this->memfree(); }
};