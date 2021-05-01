#include <stdio.h>
#include <string.h>
#include <chrono>
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

void extractLU(double* op, double* l, double* u, size_t size) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      if(i == j){
        l[i * size + j] = 1;
        u[i * size + j] = op[i * size + j];
      }
      else if (i > j) {
        l[i * size + j] = op[i * size + j];
        u[i * size + j] = 0;
      }
      else {
        l[i * size + j] = 0;
        u[i * size + j] = op[i * size + j];
      }
    }
  }
}

void luSequential(double* opMatrix, double* resMatrix, size_t size) {
  for (size_t k = 0; resMatrix[k * size + k] != 0 && k < size; k++) {
    for (size_t i = k + 1; i < size; i++) {
      resMatrix[i * size + k] /= resMatrix[k * size + k];
    }

    for (size_t i = k + 1; i < size; i++) {
      for (size_t j = k + 1; j < size; j++) {
        resMatrix[i * size + j] -=
            resMatrix[i * size + k] * resMatrix[k * size + j];
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
  // int blockSize = argc == 5 ? atoi(argv[4]) : matrixSize * matrixSize;

  const int MATRIX_SIZE_BYTES = (matrixSize * matrixSize) * sizeof(double);

  // init matrices
  double* opMatrix = (double*)malloc(MATRIX_SIZE_BYTES);
  double* resMatrix = (double*)malloc(MATRIX_SIZE_BYTES);

  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < matrixSize; j++) {
      // Diagonal can't be zero
      do {
        opMatrix[i * matrixSize + j] = (double)getRandBetween(-10, 10);
      } while(i == j && opMatrix[i * matrixSize + j] == 0);

      resMatrix[i * matrixSize + j] = 0;
    }
  }

  for (int i = 0; i < runs; i++) {

    // We do this here instead of inside the functions to avoid affecting the 
    // times of execution
    memcpy(resMatrix, opMatrix, matrixSize * matrixSize * sizeof(double));

    // Start counting
    auto begin = std::chrono::high_resolution_clock::now();
    switch (op) {
      case 1:
        luSequential(opMatrix, resMatrix, matrixSize);
        break;
    }
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::microseconds>(end - begin);

    cout << op << " " << matrixSize << " " << elapsed.count()/ 1000000.0 << endl;
  }

  free(opMatrix);
  free(resMatrix);
}