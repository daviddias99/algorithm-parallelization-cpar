#include <iostream>
#include "lu_seq.h"
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

    // Do LU factorization of block A00
    luSequential(diagonalBlock, blockSize, blockSize, size);

    if (size - currentDiagonalIdx <= blockSize) break;

    // Do LU factorization of block A10
    double *a10 = diagonalBlock + size * blockSize;

    for (size_t ii = 0; ii < size - currentDiagonalIdx - blockSize;
         ii += blockSize) {
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