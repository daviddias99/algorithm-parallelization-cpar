#include <iostream>

#include "lu_seq.h"

/*
| A00 A01 |
| A10 A11 |
*/
void luDataParallel(double *matrix, size_t size, size_t blockSize) {
#pragma omp parallel num_threads(8)
  {
    double *diagonalBlock = matrix;

    for (size_t currentDiagonalIdx = 0; currentDiagonalIdx < size;
         currentDiagonalIdx += blockSize) {
      // Do LU factorization of block A00
#pragma omp single
      { luSequential(diagonalBlock, blockSize, blockSize, size); }

      if (size - currentDiagonalIdx <= blockSize) break;

        // Do LU factorization for block A10
#pragma omp for nowait
      for (size_t ii = 0; ii < size - currentDiagonalIdx - blockSize;
           ii += blockSize) {
        factorizeA10(diagonalBlock, size, blockSize, ii);
      }

      // Do LU factorization for block A01
#pragma omp for schedule(dynamic)
      for (size_t jj = blockSize; jj < size - currentDiagonalIdx;
           jj += blockSize) {
        factorizeA01(diagonalBlock, size, blockSize, jj, jj + blockSize);
      }

      size_t sizeA11 = size - (blockSize + currentDiagonalIdx);

      // Update A11
#pragma omp for collapse(2)
      for (size_t ii = 0; ii < sizeA11; ii += blockSize)
        for (size_t jj = 0; jj < sizeA11; jj += blockSize)
          factorizeA11(diagonalBlock, size, blockSize, ii, jj);

      diagonalBlock += blockSize * size + blockSize;
    }
  }
}