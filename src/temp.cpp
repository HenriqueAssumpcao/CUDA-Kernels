#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>

#include "matrix.h"

// cpu
void rand_matrix(matrix &mat, const size_t n = 10, const float d = 1, const int seed=0){
    for(size_t i = 0; i < mat.get_nrows(); i++){
        for(size_t j = 0; j < mat.get_ncols(); j++){
            mat(i,j) = static_cast<float>(rand() % n)/d;
        }
    }
}
void matmul_naive(const matrix &x, const matrix &y, matrix &z){
    for(size_t i = 0; i < x.get_nrows(); i++){
        for(size_t j = 0; j < y.get_ncols(); j++){
            double dot = 0.0;
            for(size_t k = 0; k < x.get_ncols(); k++){
                dot += x(i,k)*y(k,j);
            }
            z(i,j) = dot;
        }
    }
}

void benchmark_matmul_cpu(const size_t nruns,const size_t n,const size_t m, const size_t p, const int seed){
    std::cout << "[CPU] MATMUL" << std::endl;
    matrix x(n,m),y(m,p),xpy(n,p); // all on host by default

    std::vector<float> durations(nruns);
    for(size_t crun = 0; crun < nruns; crun++){
        srand(seed+crun);
        rand_matrix(x);rand_matrix(y);

        auto start = std::chrono::high_resolution_clock::now();

        matmul_naive(x,y,xpy);

        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = stop - start;

        durations[crun] = duration.count() * 1000.0f;
    }
    double total_rt = 0.0;
    for(size_t i = 0; i < nruns; i++){
        total_rt += durations[i];
    }
    std::cout << "Total matmul time: " << total_rt << " ms" << std::endl;
    std::cout << "Average matmul time: " << total_rt/(double)nruns << " ms" << std::endl;

    x.memfree();y.memfree();xpy.memfree();
}

int main(int argc, char* argv[]){
    if(argc < 6){  // Need 6 arguments (program name + 5 params)
        std::cout << "Usage: " << argv[0] << " <num_runs> <n> <m> <p> <seed>" << std::endl;
        return 1;
    }
    
    try {
        size_t nruns = std::stoull(argv[1]);  // Use stoull for size_t
        size_t n = std::stoull(argv[2]);
        size_t m = std::stoull(argv[3]); 
        size_t p = std::stoull(argv[4]);
        int seed = std::stoi(argv[5]);
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}