#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <driver_types.h>

#include "utils.cuh"

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

/**
 * @brief Compare two matrices for numerical equivalence within a tolerance
 * @param A First matrix to compare
 * @param B Second matrix to compare
 * @param tolerance Maximum allowed absolute difference between elements
 * @return true if all corresponding elements differ by at most tolerance, false
 * otherwise
 *
 * Performs element-wise comparison of two matrices. Matrices must have
 * identical dimensions. Useful for validating correctness of matrix operations
 * with floating-point arithmetic.
 */
bool matrices_close(const matrix &A, const matrix &B, float tolerance) {
    assert(A.size == B.size);
    for (size_t i = 0; i < (A.nrows * A.ncols); ++i) {
        if (fabs(A.data[i] - B.data[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Compute statistical measures (mean and standard deviation) of an array
 * @param data Input array of floating-point values
 * @param size Number of elements in the array
 * @param mean Reference to store computed mean value
 * @param std_dev Reference to store computed standard deviation
 *
 * Calculates the arithmetic mean and population standard deviation of the input
 * data. Uses a two-pass algorithm: first pass computes mean, second pass
 * computes variance.
 */
void compute_stats(const float data[], const int size, float &mean,
                   float &std_dev) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    mean = sum / size;

    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    std_dev = sqrtf(variance / size);
}

/**
 * @brief Fill matrix with random values within a specified range
 * @param mat Matrix to populate with random values (must be in host memory)
 * @param min_val Minimum value for random generation (default: 0.0)
 * @param max_val Maximum value for random generation (default: 1.0)
 *
 * Fills the matrix with uniformly distributed random floating-point values
 * between min_val and max_val using the standard C rand() function.
 */
void rand_matrix(matrix &mat, float min_val = 0.0f, float max_val = 1.0f) {
    float range = max_val - min_val;

    for (size_t i = 0; i < (mat.nrows * mat.ncols); i++) {
        mat.data[i] = min_val + (float)rand() / RAND_MAX * range;
    }
}

/**
 * @brief Initialize all matrix elements to zero
 * @param mat Matrix to zero-initialize (must be in host memory)
 *
 * Sets all elements of the matrix to 0.0f using nested loops
 * to iterate through rows and columns.
 */
void zero_matrix(matrix &mat) {
    for (size_t i = 0; i < mat.nrows; i++) {
        for (size_t j = 0; j < mat.ncols; j++) {
            mat.data[i * mat.ncols + j] = 0;
        }
    }
}

/**
 * @brief Print matrix contents to standard output in readable format
 * @param mat Matrix to print (must be in host memory)
 *
 * Displays the matrix in a bracketed format with each row on a separate line.
 * Only works for matrices in host memory; displays an error message for
 * device matrices. Format: [[row1], [row2], ..., [rowN]]
 */
void print_matrix(const matrix &mat) {
    if (!mat.device) {
        std::cout << "[";
        for (size_t i = 0; i < mat.nrows; i++) {
            std::cout << "[";
            for (size_t j = 0; j < mat.ncols - 1; j++) {
                std::cout << mat.data[i * mat.ncols + j] << ",";
            }
            std::cout << mat.data[i * mat.ncols + (mat.ncols - 1)] << "]"
                      << std::endl;
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "Cant print matrix on device memory." << std::endl;
    }
}