#include <iostream>

#include "lu_seq.h"

/*
| A00 A01 |
| A10 A11 |
*/
void luDataParallel(double *matrix, size_t size, size_t blockSize) {
  double *diagonalBlock, *factorizedColumns, *factorizedRows, *subMatrix, *a10;
  size_t k, offsetK, offsetI, i, rowOffsetI, j, subMatrixSize,
      currentDiagonalIdx, ii, jj, rowOffsetK;

// Move along matrix diagonal
#pragma omp parallel num_threads(8) private(                                 \
    ii, jj, i, j, k, rowOffsetI, rowOffsetK, currentDiagonalIdx,             \
    diagonalBlock, offsetI, offsetK, factorizedColumns, factorizedRows, a10, \
    subMatrixSize, subMatrix)

  for (currentDiagonalIdx = 0; currentDiagonalIdx < size;
       currentDiagonalIdx += blockSize) {
    // Get current diagonal block start address (A00)
    diagonalBlock = matrix + currentDiagonalIdx * size + currentDiagonalIdx;

// Do LU factorization of block A00
#pragma omp single
    { luSequential(diagonalBlock, blockSize, blockSize, size); }

    if (size - currentDiagonalIdx <= blockSize) break;

    // Do LU factorization of block A10
    a10 = diagonalBlock + size * blockSize;

#pragma omp for nowait
    for (ii = 0; ii < size - currentDiagonalIdx - blockSize; ii += blockSize) {
      for (k = 0; k < blockSize && matrix[k * size + k] != 0; k++) {
        offsetK = k * size;
        for (i = ii; i < ii + blockSize; i++) {
          a10[i * size + k] /= diagonalBlock[offsetK + k];
        }

        for (i = ii; i < ii + blockSize; i++) {
          offsetI = i * size;
          for (j = k + 1; j < blockSize; j++) {
            a10[offsetI + j] -= a10[offsetI + k] * diagonalBlock[offsetK + j];
          }
        }
      }
    }

// TODO ver se é preciso um schedule dynamic
// Do LU factorization for block A01
#pragma omp for schedule(dynamic)
    for (jj = currentDiagonalIdx + blockSize; jj < size; jj += blockSize) {
      for (k = currentDiagonalIdx;
           matrix[k * size + k] != 0 && k < currentDiagonalIdx + blockSize;
           k++) {
        offsetK = k * size;
        for (i = k + 1; i < currentDiagonalIdx + blockSize; i++) {
          rowOffsetI = i * size;
          for (j = jj; j < jj + blockSize; j++) {
            matrix[rowOffsetI + j] -=
                matrix[rowOffsetI + k] * matrix[offsetK + j];
          }
        }
      }
    }

    // Calculate addresses of blocks A10, A01 and A11
    factorizedColumns = diagonalBlock + blockSize * size;
    factorizedRows = diagonalBlock + blockSize;
    subMatrix = diagonalBlock + blockSize * size + blockSize;

    subMatrixSize = size - (blockSize + currentDiagonalIdx);

    // Update A11
#pragma omp for collapse(2)
    for (ii = 0; ii < subMatrixSize; ii += blockSize)
      for (jj = 0; jj < subMatrixSize; jj += blockSize) {
        for (i = ii; i < ii + blockSize; i++) {
          rowOffsetI = i * size;
          for (k = 0; k < blockSize; k++) {
            rowOffsetK = k * size;
            for (j = jj; j < jj + blockSize; j++) {
              subMatrix[rowOffsetI + j] -= factorizedColumns[rowOffsetI + k] *
                                           factorizedRows[rowOffsetK + j];
            }
          }
        }
      }
  }
}