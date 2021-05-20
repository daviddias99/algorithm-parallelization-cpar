#include <math.h>
#include <chrono>
#include <iostream>

using namespace std;

#define TEST_MODE false

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
if (code != cudaSuccess)
{
fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
if (abort) exit(code);
}
}
#define TILE_WIDTH 32
__global__ void MatrixMulKernelBlockLocalMemFixed(double *Md, double *Nd, double *Pd, int width) {
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
    Mds[ty][tx] = Md[Row*width + (m*TILE_WIDTH + tx)];
    Nds[ty][tx] = Nd[Col + (m*TILE_WIDTH + ty)*width];
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k) Pvalue += Mds[ty][k] * Nds[k][tx];
    __syncthreads();
  }

  Pd[Row*width+Col] = Pvalue; 
}

__global__ void MatrixMulKernelBlockLocalMem(double *Md, double *Nd, double *Pd, int width, int blockSize) {
  __shared__ double* Mds;
  __shared__ double* Nds;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // Identify the row and column of the Pd element to work on
  int Row = by * blockSize + ty;
  int Col = bx * blockSize + tx;
  double Pvalue = 0;

  if (tx == 0 && ty == 0) {
    Mds = (double*) malloc(blockSize * blockSize * sizeof(double));
    Nds = (double*) malloc(blockSize * blockSize * sizeof(double));
  }
  __syncthreads();
  // Loop over the Md and Nd tiles required to compute the Pd element
  for (int m = 0; m < width / blockSize; ++m) {
    // Coolaborative loading of Md and Nd tiles into shared memory
    Mds[ty*blockSize + tx] = Md[Row*width + (m*blockSize + tx)];
    Nds[ty*blockSize + tx] = Nd[Col + (m*blockSize + ty)*width];
    __syncthreads();
    for (int k = 0; k < blockSize; ++k) Pvalue +=  Mds[ty*blockSize + k] * Nds[k*blockSize + tx];
    __syncthreads();
  }
  __syncthreads();
  // Only one thread may free the memory!
  if (tx == 0 && ty == 0) {
    free(Mds);
    free(Nds);
  }
  Pd[Row*width+Col] = Pvalue; 
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void MatrixMulKernelBlock(double* Md, double* Nd, double* Pd, int Width, int blockSize)
{
  // Calculate the row index of the Pd element and M
  int Row = blockIdx.y*blockSize + threadIdx.y;
  // Calculate the column idenx of Pd and N
  int Col = blockIdx.x*blockSize + threadIdx.x;
  double Pvalue = 0;
  // each thread computes one element of the block sub-matrix
  for (int k = 0; k < Width; ++k)
    Pvalue += Md[Row * Width + k] * Nd[k * Width + Col];
  Pd[Row * Width + Col] = Pvalue;
} 

int main(int argc, char *argv[]) {

  // Get arguments
  int matrixSize = atoi(argv[1]);
  int op = atoi(argv[2]);
  int runs = atoi(argv[3]);
  int blockSize = atoi(argv[4]);

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
  dim3 dimGrid(matrixSize/blockSize, matrixSize/blockSize);
  dim3 dimBlock(blockSize, blockSize); 

  if(op == 2)
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,matrixSize * matrixSize * sizeof(double));

  for(int i = 0; i < runs; i++) {
    // Start counting
    auto begin = std::chrono::steady_clock::now();

    switch(op) {
      case 1:
        MatrixMulKernelBlockLocalMemFixed<<<dimGrid, dimBlock>>>(Md, Nd, Pd, matrixSize);
        break;
      case 2:
        MatrixMulKernelBlockLocalMem<<<dimGrid, dimBlock>>>(Md, Nd, Pd, matrixSize, blockSize);
        break;
      case 3:
        MatrixMulKernelBlock<<<dimGrid, dimBlock>>>(Md, Nd, Pd, matrixSize, blockSize);
        break;
    }

    cudaDeviceSynchronize(); 

    auto end = std::chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<std::chrono::microseconds>(end - begin);
    cout << 1 << " " << matrixSize << " " << blockSize << " " << elapsed.count()/ 1000000.0  << " N/A" << endl;
    
    if(TEST_MODE){
      gpuErrchk( cudaPeekAtLastError() );
      cudaMemcpy(P, Pd, matrixSize*matrixSize * sizeof(double), cudaMemcpyDeviceToHost);
      cout << P[(matrixSize*(matrixSize-1) + matrixSize - 1)] << endl;
      float flops =(2.0f * matrixSize * matrixSize * matrixSize / (elapsed.count() / 1000000.0f)) * 1.0e-9f;
      cout << flops << endl;
    }
  }


  cudaFree(Md);
  cudaFree(Nd);
  cudaFree(Pd);
}