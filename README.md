# Repos name

**2020/2021** - 4th Year, 2nd Semester

**Course:** *Computação Paralela* ([CPAR](https://sigarra.up.pt/feup/pt/ucurr_geral.ficha_uc_view?pv_ocorrencia_id=350413)) | Parallel Computing

**Authors:** David Silva ([daviddias99](https://github.com/daviddias99)), Luís Cunha ([luispcunha](https://github.com/luispcunha))

---

**Description:** For the second project we talked algorithm parallelization. To do this we utilized different algorithms: matrix multiplication and LU factorization. We utilized different frameworks to achieve parallelism: OpenMP (CPU), Sycl (CPU and GPU) and CUDA (baseline for GPU). We measured the performance of these algorithms for different matrix and block sizes.

For info on the proposed work on `docs/specification.pdf` and on our results in `docs/report.pdf`.

**Technologies:** C/C++, OpenMP, Sycl, CUDA

**Skills:** Algorithm parallelization, algorithms, cache efficiency, matrix multiplication, blocking approach

**Grade:** 18.4/20

**Previous work:** [Link for the first project](https://github.com/daviddias99/matrix-multiplication-optimization-feup-cpar)

---

## Instructions

This project uses OpenMP, CUDA and Sycl. For the Sycl implementation we mainly used ComputeCpp, however, due to the lack of support of CUDA devices we used the DPC++ implementation to compile the Sycl code for an Nvidia GTX 1060.

## Compiling and running

You can run the `compile_all.py` script, or if you want to compile them individually:

### OpenMP

To compile the OpenMP code simple use the following command:

`g++ -O2 <file_name> -fopenmp` 

Both the matrix multiplication and the LU decomposition OMP programs receive the following arguments: **matrix size**, **operation**, **number of runs** and **block size**.

### CUDA

To compile the CUDA version use the following command.

`nvcc <file_name>`

The CUDA program receives the following arguments: **matrix size**, **operation**, **number of runs** and **block size**.

### Sycl

If the goal is to compile the code to the CPU or to a GPU that is supported by ComputeCpp then we just need to execute the following command on a folder that contains a `Makefile`:

`make <file_name_no_extension>`

However, if we need to compile the code for a CUDA GPU we must have the DPC++ implementation of Sycl and we can use the following command: 

`clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice <file_name>`

The Sycl program receives the following arguments: **matrix size**, **block size**, **operation**, **device** (gpu, cpu or manual to interactively choose the device), **number of runs** and **block size**.

## Program algorithm options

`cuda/matmul.cu`

1 - Matrix multiplication using blocks and local memory (static array creation), block size defined by the `TILE_WIDTH` macro.

2 - Matrix multiplication using blocks and local memory (dynamic array creation)

3 - Matrix multiplication using blocks

`omp/lu/lu.cpp`

1 - LU naive

2 - LU blocks

3 - LU OMP Data Parallel

4 - LU OMP Functional Parallel

`omp/matmul/matmul.cpp`

1 - MM naive sequential

2 - MM OMP without collapse directive

3 - MM OMP with collapse directive

`sycl/lu/lu.cpp`

1 - LU factorization

`sycl/matmul/matmul.cpp`

1 - MM naive

2 - MM Blocks without local memory

3 - MM Blocks with local memory