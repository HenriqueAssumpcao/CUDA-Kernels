#pragma once

#include <stdlib.h>

#include <assert.h>
#include <cmath>
#include <iostream>

/**
 * @brief CPU implementation of single-precision general matrix multiplication
 * (SGEMM)
 * @param A Input matrix A (M x K) in row-major order
 * @param B Input matrix B (K x N) in row-major order
 * @param C Output matrix C (M x N) in row-major order
 * @param M Number of rows in matrices A and C
 * @param K Number of columns in A and rows in B (shared dimension)
 * @param N Number of columns in matrices B and C
 * @param alpha Scalar multiplier for A*B product (default: 1.0)
 * @param beta Scalar multiplier for existing C values (default: 0.0)
 *
 * Performs the operation C = alpha * A * B + beta * C using a triple nested
 * loop. This serves as a reference implementation for validating GPU kernels.
 */
void sgemm_cpu(const float *A, const float *B, float *C, int M, int K, int N,
               float alpha = 1, float beta = 0) {

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double dot = 0.0;
            for (int k = 0; k < K; ++k) {
                dot += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * dot + beta * C[i * N + j];
        }
    }
}

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