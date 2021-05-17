#ifndef LU_BLOCKS_CMN
#define LU_BLOCKS_CMN

#include <iostream>

/**
 * Factorize the block of A10 (rows below diagonalBlock) starting at ii
 */
void factorizeA10(double *diagonalBlock, size_t size, size_t blockSize,
                  size_t ii) {
  double *a10 = diagonalBlock + size * blockSize;

  for (size_t k = 0; k < blockSize; k++) {
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

void factorizeA01(double *diagonalBlock, size_t size, size_t blockSize,
                  size_t jj, size_t end) {
  // Do LU factorization for block A01
  for (size_t k = 0; k < blockSize; k++) {
    size_t offsetK = k * size;
    for (size_t i = k + 1; i < blockSize; i++) {
      size_t rowOffsetI = i * size;
      for (size_t j = jj; j < end; j++) {
        diagonalBlock[rowOffsetI + j] -=
            diagonalBlock[rowOffsetI + k] * diagonalBlock[offsetK + j];
      }
    }
  }
}

void factorizeA11(double *diagonalBlock, size_t size, size_t blockSize,
                  size_t ii, size_t jj) {
  // Calculate addresses of blocks A10, A01 and A11
  double *factorizedColumns = diagonalBlock + blockSize * size;
  double *factorizedRows = diagonalBlock + blockSize;
  double *subMatrix = diagonalBlock + blockSize * size + blockSize;

  for (size_t i = ii; i < ii + blockSize; i++) {
    size_t rowOffsetI = i * size;
    for (size_t k = 0; k < blockSize; k++) {
      size_t rowOffsetK = k * size;
      for (size_t j = jj; j < jj + blockSize; j++) {
        subMatrix[rowOffsetI + j] -=
            factorizedColumns[rowOffsetI + k] * factorizedRows[rowOffsetK + j];
      }
    }
  }
}

#endif
