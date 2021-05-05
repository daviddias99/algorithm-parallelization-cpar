#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <iomanip>
#include <iostream>

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

void luSequential(double *matrix, size_t nRows, size_t nCols,
                  size_t matrixSize) {
  for (size_t k = 0; k < nCols && matrix[k * matrixSize + k] != 0; k++) {
    size_t offsetK = k * matrixSize;

    for (size_t i = k + 1; i < nRows; i++) {
      matrix[i * matrixSize + k] /= matrix[offsetK + k];
    }

    for (size_t i = k + 1; i < nRows; i++) {
      size_t offsetI = i * matrixSize;
      for (size_t j = k + 1; j < nCols; j++) {
        matrix[offsetI + j] -= matrix[offsetI + k] * matrix[offsetK + j];
      }
    }
  }
}

/**
 * | A00 A01 |
 * | A10 A11 |
 */
void luBlocks(double *matrix, size_t size, size_t blockSize) {
  // Move along matrix diagonal
  for (size_t currentDiagonalIdx = 0; currentDiagonalIdx < size;
       currentDiagonalIdx += blockSize) {
    // Get current diagonal block start address (A00)
    double *diagonalBlock =
        matrix + currentDiagonalIdx * size + currentDiagonalIdx;

    // cout << "---" << endl;
    // printMatrix(resMatrix, size);

    // Do LU factorization of block A00 and A10
    luSequential(diagonalBlock, size - currentDiagonalIdx, blockSize, size);

    if (size - currentDiagonalIdx <= blockSize) break;

    // Do LU factorization for block A01
    for (size_t k = currentDiagonalIdx;
         matrix[k * size + k] != 0 && k < currentDiagonalIdx + blockSize; k++) {
      size_t offsetK = k * size;
      for (size_t i = k + 1; i < currentDiagonalIdx + blockSize; i++) {
        size_t rowOffsetI = i * size;
        for (size_t j = currentDiagonalIdx + blockSize; j < size; j++) {
          matrix[rowOffsetI + j] -=
              matrix[rowOffsetI + k] * matrix[offsetK + j];
        }
      }
    }

    // Calculate addresses of blocks A10, A01 and A11
    double *factorizedColumns = diagonalBlock + blockSize * size;
    double *factorizedRows = diagonalBlock + blockSize;
    double *subMatrix = diagonalBlock + blockSize * size + blockSize;

    // Size of submatrix to update
    size_t subMatrixSize = size - (blockSize + currentDiagonalIdx);

    // Update A11
    for (size_t ii = 0; ii < subMatrixSize; ii += blockSize)
      for (size_t jj = 0; jj < subMatrixSize; jj += blockSize)
        for (size_t i = ii; i < ii + blockSize; i++) {
          size_t rowOffsetI = i * size;
          for (size_t k = 0; k < blockSize; k++) {
            size_t rowOffsetK = k * size;
            for (size_t j = jj; j < jj + blockSize; j++) {
              subMatrix[rowOffsetI + j] -= factorizedColumns[rowOffsetI + k] *
                                           factorizedRows[rowOffsetK + j];
            }
          }
        }
  }
}

/*
| A00 A01 |
| A10 A11 |
*/

void luOpenMP(double *resMatrix, size_t size, size_t blockSize) {
  double *currentMatrixPosition, *op1Matrix, *op2Matrix, *resultMatrix;
  size_t k, offsetK, i, rowOffsetI, j, sizeLimit, currentDiagonalBlock, ii, jj,
      rowOffsetK;

// Move along matrix diagonal
#pragma omp parallel num_threads(8) private(                                  \
    ii, jj, i, j, k, rowOffsetI, rowOffsetK, sizeLimit, currentDiagonalBlock, \
    currentMatrixPosition)
  for (currentDiagonalBlock = 0; currentDiagonalBlock < size;
       currentDiagonalBlock += blockSize) {
    // Get current diagonal block start address (A00)
    currentMatrixPosition =
        resMatrix + currentDiagonalBlock * size + currentDiagonalBlock;

// cout << "---" << endl;
// printMatrix(resMatrix, size);

// Do LU factorization of block A00 and A10
#pragma omp single
    {
      luSequential(currentMatrixPosition, size - currentDiagonalBlock,
                   blockSize, size);

      // Do LU factorization for block A01
      for (k = currentDiagonalBlock;
           resMatrix[k * size + k] != 0 && k < currentDiagonalBlock + blockSize;
           k++) {
        offsetK = k * size;
        for (i = k + 1; i < currentDiagonalBlock + blockSize; i++) {
          rowOffsetI = i * size;
          for (j = currentDiagonalBlock + blockSize; j < size; j++) {
            resMatrix[rowOffsetI + j] -=
                resMatrix[rowOffsetI + k] * resMatrix[offsetK + j];
          }
        }
      }
    }

    if (size - currentDiagonalBlock <= blockSize) break;

#pragma omp barrier

    // Calculate addresses of blocks A10, A01 and A11
    op1Matrix = currentMatrixPosition + blockSize * size;
    op2Matrix = currentMatrixPosition + blockSize;
    resultMatrix = currentMatrixPosition + blockSize * size + blockSize;

    sizeLimit = size - (blockSize + currentDiagonalBlock);
// Update A11
#pragma omp for collapse(2)
    for (ii = 0; ii < sizeLimit; ii += blockSize)
      for (jj = 0; jj < sizeLimit; jj += blockSize) {
        for (i = ii; i < ii + blockSize; i++) {
          rowOffsetI = i * size;
          for (k = 0; k < blockSize; k++) {
            rowOffsetK = k * size;
            for (j = jj; j < jj + blockSize; j++) {
              resultMatrix[rowOffsetI + j] -=
                  op1Matrix[rowOffsetI + k] * op2Matrix[rowOffsetK + j];
            }
          }
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

  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < matrixSize; j++) {
      // Diagonal can't be zero
      do {
        resMatrix[i * matrixSize + j] = (double)getRandBetween(-10, 10);
      } while (i == j && resMatrix[i * matrixSize + j] == 0);
    }
  }

  // cout << "-----------Original-------------" << endl;
  // printMatrix(resMatrix, matrixSize);

  for (int i = 0; i < runs; i++) {
    memcpy(opMatrix, resMatrix, MATRIX_SIZE_BYTES);
    // We do this here instead of inside the functions to avoid affecting the
    // times of execution
    // Start counting
    auto begin = std::chrono::high_resolution_clock::now();
    switch (op) {
      case 1:
        luSequential(opMatrix, matrixSize, matrixSize, matrixSize);
        // cout << "-----------Solved-------------" << endl;
        // printMatrix(opMatrix, matrixSize);
        break;
      case 2:
        memcpy(opMatrix, resMatrix, MATRIX_SIZE_BYTES);
        luBlocks(opMatrix, matrixSize, blockSize);
        // cout << "-----------Solved-------------" << endl;
        // printMatrix(opMatrix, matrixSize);
        break;
      case 3:
        memcpy(opMatrix, resMatrix, MATRIX_SIZE_BYTES);
        luOpenMP(opMatrix, matrixSize, blockSize);
        // cout << "-----------Solved-------------" << endl;
        // printMatrix(opMatrix, matrixSize);
        break;
    }
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::microseconds>(end - begin);
    cout << "-----------.-------------" << endl;
    // printMatrix(opMatrix, matrixSize);
    cout << op << " " << matrixSize << " " << elapsed.count() / 1000000.0
         << endl;
  }

  free(opMatrix);
  free(resMatrix);

  return 0;
}