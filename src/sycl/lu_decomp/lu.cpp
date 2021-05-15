#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#include "lu_seq.h"

using namespace cl::sycl;
using namespace std;

class lu_kernel;

int getRandBetween(int min, int max) { return rand() % (max - min + 1) + min; }

void outputDevInfo(const sycl::device& dev) {
  std::cout << "  -> Selected device: "
            << dev.get_info<sycl::info::device::name>() << std::endl;
  std::cout << "  -> Device vendor: "
            << dev.get_info<sycl::info::device::vendor>() << std::endl;
}

void printMatrix(double* matrix, size_t size) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      std::cout << std::setw(12) << matrix[i * size + j];
    }
    std::cout << std::endl;
  }
}

bool luFactorization(double* MA, size_t matSize, size_t blockSize,
                     const device_selector& selector) {
  queue Q(selector, [&](exception_list eL) {
    try {
      for (auto& e : eL) {
        std::rethrow_exception(e);
      }
    } catch (cl::sycl::exception e) {
      std::cout << " An exception has been thrown: " << e.what() << std::endl;
    }
  });
  double *diagonalBlock, *factorizedColumns, *factorizedRows, *subMatrix, *a10;
  size_t k, offsetK, offsetI, i, rowOffsetI, j, subMatrixSize,
      currentDiagonalIdx, ii, jj, rowOffsetK;

  for (currentDiagonalIdx = 0; currentDiagonalIdx < matSize;
       currentDiagonalIdx += blockSize) {
    // Get current diagonal block start address (A00)
    diagonalBlock = MA + currentDiagonalIdx * matSize + currentDiagonalIdx;

    // Do LU factorization of block A00
    { luSequential(diagonalBlock, blockSize, blockSize, matSize); }

    if (matSize - currentDiagonalIdx <= blockSize) break;

    // Do LU factorization of block A10
    a10 = diagonalBlock + matSize * blockSize;

    // TODO: make both loops parallel with each other
    for (ii = 0; ii < matSize - currentDiagonalIdx - blockSize;
         ii += blockSize) {
      for (k = 0; k < blockSize && MA[k * matSize + k] != 0; k++) {
        offsetK = k * matSize;
        for (i = ii; i < ii + blockSize; i++) {
          a10[i * matSize + k] /= diagonalBlock[offsetK + k];
        }

        for (i = ii; i < ii + blockSize; i++) {
          offsetI = i * matSize;
          for (j = k + 1; j < blockSize; j++) {
            a10[offsetI + j] -= a10[offsetI + k] * diagonalBlock[offsetK + j];
          }
        }
      }
    }

    // Do LU factorization for block A01
    for (jj = currentDiagonalIdx + blockSize; jj < matSize; jj += blockSize) {
      for (k = currentDiagonalIdx;
           MA[k * matSize + k] != 0 && k < currentDiagonalIdx + blockSize;
           k++) {
        offsetK = k * matSize;
        for (i = k + 1; i < currentDiagonalIdx + blockSize; i++) {
          rowOffsetI = i * matSize;
          for (j = jj; j < jj + blockSize; j++) {
            MA[rowOffsetI + j] -= MA[rowOffsetI + k] * MA[offsetK + j];
          }
        }
      }
    }

    // // Calculate addresses of blocks A10, A01 and A11
    // factorizedColumns = diagonalBlock + blockSize * matSize;
    // factorizedRows = diagonalBlock + blockSize;
    // subMatrix = diagonalBlock + blockSize * matSize + blockSize;

    // subMatrixSize = matSize - (blockSize + currentDiagonalIdx);

    // // Update A11
    // for (ii = 0; ii < subMatrixSize; ii += blockSize)
    //   for (jj = 0; jj < subMatrixSize; jj += blockSize) {
    //     for (i = ii; i < ii + blockSize; i++) {
    //       rowOffsetI = i * matSize;
    //       for (k = 0; k < blockSize; k++) {
    //         rowOffsetK = k * matSize;
    //         for (j = jj; j < jj + blockSize; j++) {
    //           subMatrix[rowOffsetI + j] -= factorizedColumns[rowOffsetI + k]
    //           *
    //                                        factorizedRows[rowOffsetK + j];
    //         }
    //       }
    //     }
    //   }

    subMatrixSize = matSize - (blockSize + currentDiagonalIdx);

    std::cout << "SubMatSize " << subMatrixSize << std::endl;

    range<2> dimensions((matSize), (matSize));
    const property_list props = {};
    buffer<double, 2> matrix(MA, dimensions, props);

    Q.submit([&](handler& h) {
      auto matrixAcc = matrix.template get_access<access::mode::read_write>(h);

      h.parallel_for<lu_kernel>(
          nd_range<2>{range<2>(subMatrixSize, subMatrixSize),
                      range<2>(blockSize, blockSize)},
          [=](nd_item<2> item) {
            int j = item.get_global_id(1) + blockSize + currentDiagonalIdx;
            int i = item.get_global_id(0) + blockSize + currentDiagonalIdx;
            double tmp = 0.0;

            // printf("Global item (%ld,%ld)!\n", item.get_global_id(0),
            //        item.get_global_id(1));

            for (int k = currentDiagonalIdx; k < currentDiagonalIdx + blockSize;
                 k++) {
              printf("i,j,k (%ld,%ld,%ld)!\n", i, j, k);

              tmp += matrixAcc[i][k] * matrixAcc[k][j];
            }
            matrixAcc[i][j] -= tmp;
          });
    });
    Q.wait_and_throw();
  }

  return false;
}

void usage(std::string programName) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: " << std::endl;
  std::cout << programName << " [matrix size] [gpu|cpu]" << std::endl;
  std::cout << "[matrix size] : Size of the matrix to multiply (minimum 32)"
            << std::endl;
  std::cout << "[gpu|cpu]    : Use GPU or CPU device."
            << " Default is to use both " << std::endl;
}

bool runExperiment(double* MA, size_t matSize, size_t blockSize,
                   const device_selector& selector) {
  auto start = std::chrono::steady_clock::now();
  bool error = luFactorization(MA, matSize, blockSize, selector);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();

  std::cout << "Time: " << time << std::endl;
  float flops =
      (2.0f * matSize * matSize * matSize / (time / 1000.0f)) * 1.0e-9f;
  std::cout << "TODO: calcular isto direito: GFLOPs: " << flops << std::endl;

  return error;
}

bool compareResults(double* control, double* result, size_t matrixSize) {
  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < matrixSize; j++) {
      if (control[i * matrixSize + j] - result[i * matrixSize + j] > 1e-8) {
        std::cout << "ALGORITHM NOT CORRECT " << result[i * matrixSize + j]
                  << " != " << control[i * matrixSize + j] << std::endl;

        return true;
      }
    }
  }

  return false;
}

int main(int argc, char* argv[]) {
  bool gpu = true;
  bool cpu = true;
  bool error = false;

  if (argc != 3 && argc != 4) {
    usage(argv[0]);
    return 1;
  }

  size_t matSize = 0;
  try {
    matSize = std::stoi(argv[1]);
  } catch (...) {
    usage(argv[0]);
    return 1;
  }

  srand(time(NULL));

  if (matSize < 3) {
    usage(argv[0]);
    return 1;
  }

  size_t blockSize = 0;
  try {
    blockSize = std::stoi(argv[2]);
  } catch (...) {
    usage(argv[0]);
    return 1;
  }

  if (argc == 4) {
    if (std::string(argv[3]) == "gpu") {
      gpu = true;
      cpu = false;
    } else if (std::string(argv[3]) == "cpu") {
      gpu = false;
      cpu = true;
    } else {
      usage(argv[0]);
      return 1;
    }
  }

  double* originalMatrix = new double[matSize * matSize];
  double* MA = new double[matSize * matSize];
  double* controlMatrix = new double[matSize * matSize];

  for (size_t i = 0; i < matSize; i++) {
    for (size_t j = 0; j < matSize; j++) {
      do {
        originalMatrix[i * matSize + j] = (double)getRandBetween(-10, 10);
      } while (i == j && originalMatrix[i * matSize + j] == 0);
    }
  }

  if (matSize < 128) {
    memcpy(controlMatrix, originalMatrix, matSize * matSize * sizeof(double));
    luSequential(controlMatrix, matSize, matSize, matSize);
  }
  std::cout << "***** Original" << std::endl;
  printMatrix(originalMatrix, matSize);
  std::cout << "***** Control" << std::endl;
  printMatrix(controlMatrix, matSize);

  if (gpu) {
    std::cout << "***** GPU" << std::endl;
    memcpy(MA, originalMatrix, matSize * matSize * sizeof(double));

    error = runExperiment(MA, matSize, blockSize, gpu_selector{});
    // printMatrix(MA, matSize);
    if (matSize < 128 && !error) {
      std::cout << "** Result" << std::endl;
      printMatrix(MA, matSize);

      error = compareResults(controlMatrix, MA, matSize);
    }

    std::cout << (error ? "Error in computation." : "Success") << std::endl;
  }
  if (cpu) {
    std::cout << "***** CPU" << std::endl;
    memcpy(MA, originalMatrix, matSize * matSize * sizeof(double));

    error = runExperiment(MA, matSize, blockSize, cpu_selector{});
    if (matSize < 128 && !error) {
      std::cout << "** Result" << std::endl;
      printMatrix(MA, matSize);
      error = compareResults(controlMatrix, MA, matSize);
    }

    std::cout << (error ? "Error in computation." : "Success") << std::endl;
  }

  delete[] MA;

  return error ? 1 : 0;
}
