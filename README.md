# CUDA kernels
This repository is an ongoing archive of CUDA kernels I've been implementing.

The kernels can be found in the ```include/kernel/``` folder. The benchmarks can be found in the ```src/``` folder. First, use make to compile, and then run the executables in the ```build/``` directory.

Kernels in this repo:
1. The ```include/gemm/``` folder contains a naive, a blocktiled and a threadtiled implementations of GEneral Matrix Multiply (GEMM), in both FP32 and FP16 formats (SGEMM and HGEMM, respectively).
2. The ```include/attn/``` folder contains a kernel for the transpose operation, for the softmax operation, and for the flash attention forward pass, all currently in FP32.

Next steps:
1. Add warptiling, vectorized loads and double buffering.
2. Improve benchmarking.
3. Add FP16 support to flash attention.

My main reference so far has been this [incredibly useful tutorial](https://siboehm.com/articles/22/CUDA-MMM).