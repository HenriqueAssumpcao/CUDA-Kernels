#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>

// matrix of element of type T
struct dmat{
    size_t nrows,ncols;
    bool major_axis; // 0 for row major, 1 for column major
    double *arr;

    dmat(){
        this->nrows = 0;this->ncols = 0;
        this->major_axis = 0;
        this->arr = nullptr;
    }

    dmat(const size_t nrows, const size_t ncols, bool major_axis=0){
        this->nrows = nrows;this->ncols = ncols;
        this->major_axis = major_axis;
        this->arr = (double*)malloc(nrows*ncols*sizeof(double));
    }

    size_t map(const size_t r, const size_t c){
        if(major_axis == 0){
            return (r*this->ncols) + c;
        }
        else{
            return (c*this->nrows) + r;
        }
    }

    double get(const size_t r, const size_t c){
        return this->arr[this->map(r,c)];
    }

    void set(const size_t r, const size_t c, const double val){
        this->arr[this->map(r,c)] = val;
    }

    ~dmat(){
        free(this->arr);
        this->arr = nullptr;
    }
};

// utils
void print_dmat(dmat &x){
    std::cout << "[";
    for(int i = 0; i < x.nrows; i++){
        std::cout << "[";
        for(int j = 0; j < x.ncols-1; j++){
            std::cout << x.get(i,j) << ",";
        }
        std::cout << x.get(i,x.ncols-1) << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}

void init_rand_dmat(dmat &x, const size_t n = 10, const double d = 1){
    for(size_t i = 0; i < x.nrows; i++){
        for(size_t j = 0; j < x.ncols; j++){
            x.set(i,j,static_cast<double>(rand() % n)/d);
        }
    }
}

// operations
void transpose(dmat &x, dmat &x_t){
    for(size_t i = 0; i < x.nrows; i++){
        for(size_t j = 0; j < x.ncols; j++){
            x_t.set(j,i,x.get(i,j));
        }
    }
}

void matmul(dmat &x, dmat &y, dmat &z){
    for(size_t i = 0; i < x.nrows; i++){
        for(size_t j = 0; j < y.ncols; j++){
            double dot = 0;
            for(size_t k = 0 ; k < x.ncols; k++){
                dot += x.get(i,k)*y.get(k,j);
            }
            z.set(i,j,dot);
        }
    }
}

// benchmark
std::vector<double> benchmark_matmul(const size_t nruns,const size_t n,const size_t m, const int seed){
    std::vector<double> durations(nruns);
    for(size_t crun = 0; crun < nruns; crun++){
        srand(seed+crun);
        dmat x(n,m),y(m,n),xpy(n,n);
        init_rand_dmat(x);init_rand_dmat(y);

        auto start = std::chrono::high_resolution_clock::now();
        matmul(x,y,xpy);

        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = stop - start;

        durations[crun] = duration.count() * 1000.0f;
    }

    std::cout << "Average matmul time: " << std::accumulate(durations.begin(),durations.end(),0)/(double)nruns << " ms" << std::endl;
    return durations;
}

int main(int argc, char* argv[]){

    if(argc < 5){
        std::cout << "Expected (num_runs,n,m,seed) as argv" << std::endl;
        return 1;
    }

    size_t nruns = (size_t)std::stoi(argv[1]), n = (size_t)std::stoi(argv[2]),m = (size_t)std::stoi(argv[3]);
    int seed = std::stoi(argv[4]);

    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(10);

    std::vector<double> durations = benchmark_matmul(nruns,n,m,seed);

    
}

