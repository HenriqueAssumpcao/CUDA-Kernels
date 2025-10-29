# CUDA kernels
This repository is an ongoing archive of CUDA kernels I've been implementing.

The kernels can be found in the ```include/kernel/``` folder. The benchmarks can be found in the ```src/``` folder. First, use make to compile, and then run the executables in the ```build/``` directory.

Kernels currently implemented:
1. ```sgemm_naive```: naive sgemm with gmem coalescing;
2. ```sgemm_blocktiling```: sgemm with blocktiling;
3. ```sgemm_threadtiling```: sgemm with block and thread tiling;

Next steps:
1. Add warptiling, vectorized loads and double buffering.
2. Improve benchmarking.
3. Implement naive and flash attention.

My main reference so far has been this [incredibly useful tutorial](https://siboehm.com/articles/22/CUDA-MMM).