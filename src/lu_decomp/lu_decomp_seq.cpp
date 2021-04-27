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

// Falta extrair a L e a U do resultado
void seqLUClass(double* opMatrix, double* resMatrix, size_t size) {
  size_t k = 0;

  memcpy(resMatrix, opMatrix, size * size * sizeof(double));

  for(size_t k = 0;resMatrix[k * size + k] != 0 && k < size; k++ ) {
    for (size_t i = k+1; i < size; i++)
    {
      resMatrix[i*size + k] /= resMatrix[k * size + k]; 
    }
    
    for (size_t i = k+1; i < size; i++)
    {
      for (size_t j = k+1; j < size; j++)
      {
        resMatrix[i*size + j] -= resMatrix[i * size + k] * resMatrix[k*size + j]; 
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
  double* resMatrix = (double*)malloc(MATRIX_SIZE_BYTES);

  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < matrixSize; j++) {
      opMatrix[i * matrixSize + j] = (double)getRandBetween(-5, 5);
      lMatrix[i * matrixSize + j] = 0;
      uMatrix[i * matrixSize + j] = 0;
      resMatrix[i * matrixSize + j] = 0;
    }
  }


  for (int i = 0; i < runs; i++) {
    // Start counting

    double algorithmTime;

    switch (op) {
      case 1:
        seqLUClass(opMatrix, resMatrix, matrixSize);
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
    cout << "LU" << endl;
    printMatrix(resMatrix, matrixSize);

    // memset(resMatrix, 0, MATRIX_SIZE_BYTES);

    // cout << algorithmTime << endl;
  }

  free(opMatrix);
  free(lMatrix);
  free(uMatrix);
}