#include <iostream>

#include "lu_seq.h"

void factorA10(double* matrix, double* diagonalBlock, double* a10,
               size_t blockSize, size_t size, size_t ii) {
  for (size_t k = 0; k < blockSize && matrix[k * size + k] != 0; k++) {
    size_t offsetK = k * size;
    for (size_t i = ii; i < ii + blockSize; i++) {
      a10[i * size + k] /= diagonalBlock[offsetK + k];
    }

    for (size_t i = ii; i < ii + blockSize; i++) {
      size_t offsetI = i * size;
      for (size_t j = k + 1; j < blockSize; j++) {
        a10[offsetI + j] -= a10[offsetI + k] * diagonalBlock[offsetK + j];
      }
    }
  }
}

void factorA01(double* matrix, size_t jj, size_t blockSize, size_t size,
               size_t currentDiagonalIdx) {
  size_t rowOffsetI, offsetK, i, j, k;
  for (k = currentDiagonalIdx;
       matrix[k * size + k] != 0 && k < currentDiagonalIdx + blockSize; k++) {
    offsetK = k * size;
    for (i = k + 1; i < currentDiagonalIdx + blockSize; i++) {
      rowOffsetI = i * size;
      for (j = jj; j < jj + blockSize; j++) {
        matrix[rowOffsetI + j] -= matrix[rowOffsetI + k] * matrix[offsetK + j];
      }
    }
  }
}

void factorA11(double* subMatrix, double* factorizedColumns,
               double* factorizedRows, size_t size, size_t blockSize, size_t ii,
               size_t jj) {
  size_t i, j, k, rowOffsetI, rowOffsetK;
  for (i = ii; i < ii + blockSize; i++) {
    rowOffsetI = i * size;
    for (k = 0; k < blockSize; k++) {
      rowOffsetK = k * size;
      for (j = jj; j < jj + blockSize; j++) {
        subMatrix[rowOffsetI + j] -=
            factorizedColumns[rowOffsetI + k] * factorizedRows[rowOffsetK + j];
      }
    }
  }
}

/*
| A00 A01 |
| A10 A11 |
*/
void luFuncParallel(double* matrix, size_t size, size_t blockSize) {
  double *diagonalBlock, *factorizedColumns, *factorizedRows, *subMatrix, *a10;
  size_t subMatrixSize, currentDiagonalIdx, ii, jj;

// Move along matrix diagonal
#pragma omp parallel num_threads(8) private( \
    ii, jj, subMatrixSize, currentDiagonalIdx, diagonalBlock, a10)
#pragma omp single
    {
  for (currentDiagonalIdx = 0; currentDiagonalIdx < size;
       currentDiagonalIdx += blockSize) {
    // Get current diagonal block start address (A00)
    diagonalBlock = matrix + currentDiagonalIdx * size + currentDiagonalIdx;


      // Do LU factorization of block A00
      luSequential(diagonalBlock, blockSize, blockSize, size);

      if (size - currentDiagonalIdx <= blockSize) break;
 // FIXME acho que este barrier ja existe implicitamente no
                     // fim do single

      // Do LU factorization of block A10
      a10 = diagonalBlock + size * blockSize;
#pragma omp taskloop
      for (size_t ii = 0; ii < size - currentDiagonalIdx - blockSize;
           ii += blockSize) {
        factorA10(matrix, diagonalBlock, a10, blockSize, size, ii);
      }

      // Do LU factorization for block A01
#pragma omp taskloop
      for (size_t jj = currentDiagonalIdx + blockSize; jj < size;
           jj += blockSize) {
        factorA01(matrix, jj, blockSize, size, currentDiagonalIdx);
      }

      // Calculate addresses of blocks A10, A01 and A11
      factorizedColumns = diagonalBlock + blockSize * size;
      factorizedRows = diagonalBlock + blockSize;
      subMatrix = diagonalBlock + blockSize * size + blockSize;
      subMatrixSize = size - (blockSize + currentDiagonalIdx);

      // Update A11
#pragma omp taskloop collapse(2)
      for (ii = 0; ii < subMatrixSize; ii += blockSize)
        for (jj = 0; jj < subMatrixSize; jj += blockSize)
          factorA11(subMatrix, factorizedColumns, factorizedRows, size,
                    blockSize, ii, jj);
    }
  }
}