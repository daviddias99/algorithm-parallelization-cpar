#include <stdio.h>
#include <string.h>
#include <time.h>

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

// Retirada da net, falta implementar a do professor
void seqLU(double* opMatrix, double* lMatrix, double* uMatrix, size_t size) {
  int i = 0, j = 0, k = 0;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      if (j < i)
        lMatrix[j*size + i] = 0;
      else {
        lMatrix[j*size + i] = opMatrix[j*size + i];
        for (k = 0; k < i; k++) {
          lMatrix[j*size + i] = lMatrix[j*size + i] - lMatrix[j*size + k] * uMatrix[k*size + i];
        }
      }
    }
    for (j = 0; j < size; j++) {
      if (j < i)
        uMatrix[i*size + j] = 0;
      else if (j == i)
        uMatrix[i*size + j] = 1;
      else {
        uMatrix[i*size + j]  = opMatrix[i*size + j]  / lMatrix[i*size + i] ;
        for (k = 0; k < i; k++) {
          uMatrix[i*size + j]  = uMatrix[i*size + j]  - ((lMatrix[i*size + k]  * uMatrix[k*size + j] ) / lMatrix[i*size + i] );
        }
      }
    }
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
  double* lMatrix = (double*)malloc(MATRIX_SIZE_BYTES);
  double* uMatrix = (double*)malloc(MATRIX_SIZE_BYTES);

  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < matrixSize; j++) {
      opMatrix[i * matrixSize + j] = (double)getRandBetween(-5, 5);
      lMatrix[i * matrixSize + j] = 0;
      uMatrix[i * matrixSize + j] = 0;
    }
  }


  for (int i = 0; i < runs; i++) {
    // Start counting

    double algorithmTime;

    switch (op) {
      case 1:
        seqLU(opMatrix, lMatrix, uMatrix, matrixSize);
        break;
        // case 2:
        //   algorithmTime = optimCycle(op1Matrix, op2Matrix, resMatrix,
        //   matrixSize); break;
        // case 3:
        //   algorithmTime = blockCycle(op1Matrix, op2Matrix, resMatrix,
        //   matrixSize, blockSize); break;
    }
    cout << "OP" << endl;
    printMatrix(opMatrix, matrixSize);
    cout << "L" << endl;
    printMatrix(lMatrix, matrixSize);
    cout << "U" << endl;
    printMatrix(uMatrix, matrixSize);

    // memset(resMatrix, 0, MATRIX_SIZE_BYTES);

    // cout << algorithmTime << endl;
  }

  free(opMatrix);
  free(lMatrix);
  free(uMatrix);
}