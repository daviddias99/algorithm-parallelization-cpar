#include <stdio.h>
#include <string.h>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std;

int getRandBetween(int min, int max) { return rand() % (max - min + 1) + min; }

void printMatrix(double* matrix, size_t size) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      cout << setw(12) << matrix[i * size + j];
    }
    cout << endl;
  }
}

void extractLU(double* op, double* l, double* u, size_t size) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      if(i == j){
        l[i * size + j] = 1;
        u[i * size + j] = op[i * size + j];
      }
      else if (i > j) {
        l[i * size + j] = op[i * size + j];
        u[i * size + j] = 0;
      }
      else {
        l[i * size + j] = 0;
        u[i * size + j] = op[i * size + j];
      }
    }
  }
}

void luSequential(double* resMatrix, size_t lSize, size_t wSize) {
  for (size_t k = 0; resMatrix[k * lSize + k] != 0 && k < wSize; k++) {
    for (size_t i = k + 1; i < lSize; i++) {
      resMatrix[i * lSize + k] /= resMatrix[k * lSize + k];
    }

    for (size_t i = k + 1; i < lSize; i++) {
      for (size_t j = k + 1; j < wSize; j++) {
        resMatrix[i * lSize + j] -=
            resMatrix[i * lSize + k] * resMatrix[k * lSize + j];
      }
    }
  }
}

void blockCycle(double* op1Matrix, double* op2Matrix, double* resMatrix,
                  int matrixSize, int blockSize) {
  int ii, jj, kk, i, j, k, rowOffsetI, rowOffsetK;

  for (ii = 0; ii < matrixSize; ii += blockSize)
    for (jj = 0; jj < matrixSize; jj += blockSize)
      for (kk = 0; kk < matrixSize; kk += blockSize)
        for (i = ii; i < ii + blockSize; i++){
          rowOffsetI = i * matrixSize;
          for (k = kk; k < kk + blockSize; k++){
            rowOffsetK = k * matrixSize;
            for (j = jj; j < jj + blockSize; j++)
              resMatrix[rowOffsetI + j] +=
                  op1Matrix[rowOffsetI + k] *
                  op2Matrix[rowOffsetK + j];
          }
        }
}

void luBlockBased(double* resMatrix, size_t size, size_t blockSize) {
  for (size_t currentDiagonalBlock = 0; currentDiagonalBlock < size; currentDiagonalBlock += blockSize){
    double* subMatrix = resMatrix + currentDiagonalBlock * size + currentDiagonalBlock;
    luSequential(subMatrix, size - currentDiagonalBlock, blockSize);
    cout << "-----------First Step (Update LUd submatrix and columns)-------------" << endl;
    printMatrix(resMatrix, size);
    for (size_t k = currentDiagonalBlock; resMatrix[k * size + k] != 0 && k < currentDiagonalBlock + blockSize; k++) {
      for (size_t i = k + 1; i < currentDiagonalBlock + blockSize; i++) {
        for (size_t j = currentDiagonalBlock + blockSize; j < size; j++) {
          resMatrix[i * size + j] -=
              resMatrix[i * size + k] * resMatrix[k * size + j];
        }
      }
    }
    cout << "-----------Second Step (Update rows)-------------" << endl;
    printMatrix(resMatrix, size);
    for (size_t ii = currentDiagonalBlock + blockSize; ii < size; ii += blockSize)
      for (size_t jj = currentDiagonalBlock + blockSize; jj < size; jj += blockSize)
        for (size_t kk = currentDiagonalBlock + blockSize; kk < size; kk += blockSize)
          for (size_t i = ii; i < ii + blockSize; i++){
            for (size_t k = kk; k < kk + blockSize; k++){
              for (size_t j = jj; j < jj + blockSize; j++)
                resMatrix[i * size + j] -=
                    op1Matrix[i * size + k] *
                    op2Matrix[k * size + j];
            }
          }
    break;
  }
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    cerr << "ERROR: insufficient number of arguments (square matrix size, "
            "runs, operation)"
         << endl;
    return -1;
  }

  // Seed RNG
  srand(time(NULL));

  // Get arguments
  int matrixSize = atoi(argv[1]);
  int runs = atoi(argv[2]);
  int op = atoi(argv[3]);
  int blockSize = argc == 5 ? atoi(argv[4]) : matrixSize * matrixSize;

  const int MATRIX_SIZE_BYTES = (matrixSize * matrixSize) * sizeof(double);

  // init matrices
  double* opMatrix = (double*)malloc(MATRIX_SIZE_BYTES);
  double* resMatrix = (double*)malloc(MATRIX_SIZE_BYTES);

  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < matrixSize; j++) {
      // Diagonal can't be zero
      do {
        resMatrix[i * matrixSize + j] = (double)getRandBetween(-10, 10);
      } while(i == j && resMatrix[i * matrixSize + j] == 0);
    }
  }

  cout << "-----------Original-------------" << endl;
  printMatrix(resMatrix, matrixSize);

  for (int i = 0; i < runs; i++) {
    memcpy(opMatrix, resMatrix, MATRIX_SIZE_BYTES);
    // We do this here instead of inside the functions to avoid affecting the 
    // times of execution
    // Start counting
    auto begin = std::chrono::high_resolution_clock::now();
    switch (op) {
      case 1:
        luSequential(opMatrix, matrixSize, matrixSize);
        cout << "-----------Solved-------------" << endl;
        printMatrix(opMatrix, matrixSize);
      case 2:
        memcpy(opMatrix, resMatrix, MATRIX_SIZE_BYTES);
        luBlockBased(opMatrix, matrixSize, blockSize);
        break;
    }
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::microseconds>(end - begin);
    cout << "-----------.-------------" << endl;
    // printMatrix(opMatrix, matrixSize);
    cout << op << " " << matrixSize << " " << elapsed.count()/ 1000000.0 << endl;
  }

  free(opMatrix);
  free(resMatrix);

  return 0;
}