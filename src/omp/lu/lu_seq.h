#ifndef LU_SEQ
#define LU_SEQ
#define NUM_THREADS 4

#include <iostream>

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
#endif