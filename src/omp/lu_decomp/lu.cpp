#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <iomanip>
#include <iostream>

#include "lu_blocks.h"
#include "lu_data_parallel.h"
#include "lu_func_parallel.h"
#include "lu_seq.h"
using namespace std;

int getRandBetween(int min, int max) { return rand() % (max - min + 1) + min; }

void printMatrix(double *matrix, size_t size) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      cout << setw(12) << matrix[i * size + j];
    }
    cout << endl;
  }
}

void extractLU(double *op, double *l, double *u, size_t size) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      if (i == j) {
        l[i * size + j] = 1;
        u[i * size + j] = op[i * size + j];
      } else if (i > j) {
        l[i * size + j] = op[i * size + j];
        u[i * size + j] = 0;
      } else {
        l[i * size + j] = 0;
        u[i * size + j] = op[i * size + j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cerr << "ERROR: insufficient number of arguments (square matrix size, "
            "runs, operation)"
         << endl;
    return -1;
  }

  // Seed RNGsize - blockSize
  srand(time(NULL));

  // Get arguments
  int matrixSize = atoi(argv[1]);
  int runs = atoi(argv[2]);
  int op = atoi(argv[3]);
  int blockSize = argc == 5 ? atoi(argv[4]) : matrixSize;

  const int MATRIX_SIZE_BYTES = (matrixSize * matrixSize) * sizeof(double);

  // init matrices
  double *opMatrix = (double *)malloc(MATRIX_SIZE_BYTES);
  double *resMatrix = (double *)malloc(MATRIX_SIZE_BYTES);
  double *controlMatrix = (double *)malloc(MATRIX_SIZE_BYTES);

  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < matrixSize; j++) {
      // Diagonal can't be zero
      do {
        resMatrix[i * matrixSize + j] = (double)getRandBetween(-10, 10);
      } while (i == j && resMatrix[i * matrixSize + j] == 0);
    }
  }

  for (int i = 0; i < runs; i++) {
    memcpy(opMatrix, resMatrix, MATRIX_SIZE_BYTES);
    memcpy(controlMatrix, resMatrix, MATRIX_SIZE_BYTES);
    // We do this here instead of inside the functions to avoid affecting the
    // times of execution
    // Start counting
    luSequential(controlMatrix, matrixSize, matrixSize, matrixSize);
    // cout << "-------- CONTROL --------" << endl;
    // printMatrix(controlMatrix, matrixSize);
    auto begin = std::chrono::high_resolution_clock::now();
    switch (op) {
      case 1:
        break;
      case 2:
        luBlocks(opMatrix, matrixSize, blockSize);
        break;
      case 3:
        luDataParallel(opMatrix, matrixSize, blockSize);
        break;
      case 4:
        luFuncParallel(opMatrix, matrixSize, blockSize);
        break;
    }



    // cout << "-------- RES --------" << endl;
    // printMatrix(controlMatrix, matrixSize);
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::microseconds>(end - begin);

    cout << op << " " << matrixSize << " " << blockSize << " "
         << elapsed.count() / 1000000.0 << endl;
  }

  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < matrixSize; j++) {
      if (opMatrix[i * matrixSize + j] != controlMatrix[i * matrixSize + j]) {
        cout << "ALGORITHM NOT CORRECT " << opMatrix[i * matrixSize + j]
             << " != " << controlMatrix[i * matrixSize + j] << endl;

        return -1;
      }
    }
  }

  free(opMatrix);
  free(resMatrix);

  return 0;
}