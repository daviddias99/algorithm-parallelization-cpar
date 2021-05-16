#include <math.h>
#include <chrono>
#include <iostream>

using namespace std;

#define TILE_WIDTH 32
#define TEST_MODE = true

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

  Pd[Row*width+Col] = Pvalue; 
}

__global__ void MatrixMulKernelBlock(double* Md, double* Nd, double* Pd, int Width)
{
  // Calculate the row index of the Pd element and M
  int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
  // Calculate the column idenx of Pd and N
  int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;
  double Pvalue = 0;
  // each thread computes one element of the block sub-matrix
  for (int k = 0; k < Width; ++k)
    Pvalue += Md[Row * Width + k] * Nd[k * Width + Col];
  Pd[Row * Width + Col] = Pvalue;
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
    auto begin = std::chrono::steady_clock::now();
    MatrixMulKernelBlock<<<dimGrid, dimBlock>>>(Md, Nd, Pd, matrixSize);
    cudaDeviceSynchronize(); 
    auto end = std::chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<std::chrono::microseconds>(end - begin);
    cout << 1 << " " << matrixSize << " " << TILE_WIDTH << " " << elapsed.count()/ 1000000.0 << endl;
    
    if(TEST_MODE){
      float flops =(2.0f * matrixSize * matrixSize * matrixSize / (elapsed.count() / 1000000.0f)) * 1.0e-9f;
      cout << flops << endl;
    }
  }

  cudaMemcpy(P, Pd, matrixSize*matrixSize * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(Md);
  cudaFree(Nd);
  cudaFree(Pd);
}