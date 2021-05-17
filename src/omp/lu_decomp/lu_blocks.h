#include <iostream>

#include "lu_blocks_common.h"
#include "lu_seq.h"

/**
 * | A00 A01 |
 * | A10 A11 |
 */
void luBlocks(double *matrix, size_t size, size_t blockSize) {
  // Move along matrix diagonal
  double *diagonalBlock = matrix;
  
  for (size_t currentDiagonalIdx = 0; currentDiagonalIdx < size;
       currentDiagonalIdx += blockSize) {
    // Do LU factorization of block A00
    luSequential(diagonalBlock, blockSize, blockSize, size);

    if (size - currentDiagonalIdx <= blockSize) break;

    // Do LU factorization of block A10
    for (size_t ii = 0; ii < size - currentDiagonalIdx - blockSize;
         ii += blockSize) {
      factorizeA10(diagonalBlock, size, blockSize, ii);
    }

    factorizeA01(diagonalBlock, size, blockSize, blockSize,
                 size - currentDiagonalIdx);

    // Size of submatrix to update
    size_t sizeA11 = size - (blockSize + currentDiagonalIdx);

    // Update A11
    for (size_t ii = 0; ii < sizeA11; ii += blockSize)
      for (size_t jj = 0; jj < sizeA11; jj += blockSize)
        factorizeA11(diagonalBlock, size, blockSize, ii, jj);

    diagonalBlock += blockSize * size + blockSize;
  }
}