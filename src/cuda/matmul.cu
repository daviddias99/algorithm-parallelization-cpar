#include <math.h>
#include <chrono>
#include <iostream>

using namespace std;

#define TILE_WIDTH 16

__global__ void MatrixMulKernel(double *Md, double *Nd, double *Pd, int width) {
  __shared__ double Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ double Nds[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // Identify the row and column of the Pd element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  double Pvalue = 0;
  // Loop over the Md and Nd tiles required to compute the Pd element
  for (int m = 0; m < width / TILE_WIDTH; ++m) {
    // Coolaborative loading of Md and Nd tiles into shared memory
    Mds[tx][ty] = Md[(m * TILE_WIDTH + tx) * width + Row];
    Nds[tx][ty] = Nd[Col * width + (m * TILE_WIDTH + ty)];
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k) Pvalue += Mds[tx][k] * Nds[k][ty];
    __syncthreads();
  }
}



int main(int argc, char *argv[]) {

  // Get arguments
  int matrixSize = atoi(argv[1]);
  int runs = atoi(argv[2]);

  // allocate and initialize host (CPU) memory
  double *M = (double *)malloc(matrixSize * matrixSize * sizeof(double));
  double *N = (double *)malloc(matrixSize * matrixSize * sizeof(double));
  double *P = (double *)malloc(matrixSize * matrixSize * sizeof(double));

  for (int i = 0; i < matrixSize; i++){
    for (int j = 0; j < matrixSize; j++){
      M[i * matrixSize + j] = (double)1.0;
      N[i * matrixSize + j] = (double)(i + 1);
    }
  }
  // allocate device (GPU) memory
  double *Md, *Nd, *Pd;
  cudaMalloc((void **)&Md, matrixSize * matrixSize * sizeof(double));
  cudaMalloc((void **)&Nd, matrixSize * matrixSize * sizeof(double));
  cudaMalloc((void **)&Pd, matrixSize * matrixSize * sizeof(double));
  // copy host memory to device
  cudaMemcpy(Md, M, matrixSize*matrixSize * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(Nd, N, matrixSize*matrixSize * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(Pd, P, matrixSize*matrixSize * sizeof(double), cudaMemcpyHostToDevice);
  dim3 dimGrid(matrixSize/TILE_WIDTH, matrixSize/TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH); 

  for(int i = 0; i < runs; i++) {
    // Start counting
    auto begin = std::chrono::high_resolution_clock::now();
    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, matrixSize);

    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::microseconds>(end - begin);
    cout << 1 << " " << matrixSize << " " << TILE_WIDTH << " " << elapsed.count()/ 1000000.0 << endl;
  }
  cudaFree(Md);
  cudaFree(Nd);
  cudaFree(Pd);
}