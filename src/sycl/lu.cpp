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

template <typename T>
bool luFactorization(T* MA, size_t matSize, size_t blockSize,
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

  for (size_t currentDiagonalIdx = 0; currentDiagonalIdx < matSize;
       currentDiagonalIdx += blockSize) {
    for (size_t k = currentDiagonalIdx;
         k < blockSize + currentDiagonalIdx && MA[k * matSize + k] != 0; k++) {
      for (size_t i = k + 1; i < currentDiagonalIdx + blockSize; i++) {
        MA[i * matSize + k] /= MA[k * matSize + k];
      }

      for (size_t i = k + 1; i < currentDiagonalIdx + blockSize; i++) {
        for (size_t j = k + 1; j < currentDiagonalIdx + blockSize; j++) {
          MA[i * matSize + j] -= MA[i * matSize + k] * MA[k * matSize + j];
        }
      }
    }

    if (matSize - currentDiagonalIdx <= blockSize) break;

    for (size_t ii = currentDiagonalIdx; ii < matSize - blockSize;
         ii += blockSize) {
      for (size_t k = currentDiagonalIdx; k < currentDiagonalIdx + blockSize;
           k++) {
        for (size_t i = ii; i < ii + blockSize; i++) {
          MA[(i + blockSize) * matSize + k] /= MA[k * matSize + k];
        }

        for (size_t i = ii; i < ii + blockSize; i++) {
          for (size_t j = k + 1; j < currentDiagonalIdx + blockSize; j++) {
            MA[(i + blockSize) * matSize + j] -=
                MA[(i + blockSize) * matSize + k] * MA[k * matSize + j];
          }
        }
      }
    }

    for (size_t jj = currentDiagonalIdx + blockSize; jj < matSize;
         jj += blockSize) {
      for (size_t k = currentDiagonalIdx; k < currentDiagonalIdx + blockSize;
           k++) {
        for (size_t i = k + 1; i < currentDiagonalIdx + blockSize; i++) {
          for (size_t j = jj; j < jj + blockSize; j++) {
            MA[i * matSize + j] -= MA[i * matSize + k] * MA[k * matSize + j];
          }
        }
      }
    }

    for (size_t ii = currentDiagonalIdx + blockSize; ii < matSize;
         ii += blockSize)
      for (size_t jj = currentDiagonalIdx + blockSize; jj < matSize;
           jj += blockSize) {
        for (size_t i = ii; i < ii + blockSize; i++) {
          for (size_t k = currentDiagonalIdx;
               k < currentDiagonalIdx + blockSize; k++) {
            for (size_t j = jj; j < jj + blockSize; j++) {
              MA[i * matSize + j] -= MA[i * matSize + k] * MA[k * matSize + j];
            }
          }
        }
      }

    T* A_device = malloc_device<T>(
        (matSize - currentDiagonalIdx) * (matSize - currentDiagonalIdx), Q);

    Q.memcpy(A_device, &MA[currentDiagonalIdx * matSize + currentDiagonalIdx],
             (matSize - currentDiagonalIdx) * (matSize - currentDiagonalIdx) *
                 sizeof(T))
        .wait();
    Q.submit([&](handler& h) {
      h.parallel_for<lu_kernel>(
          range<2>{matSize - currentDiagonalIdx - blockSize,
                   matSize - currentDiagonalIdx - blockSize},
          [=](id<2> idx) {
            int j = blockSize + idx[0];
            int i = blockSize + idx[1];

            for (int k = currentDiagonalIdx; k < currentDiagonalIdx + blockSize;
                 ++k) {
              A_device[j * (matSize - currentDiagonalIdx) + i] -=
                  A_device[j * (matSize - currentDiagonalIdx) + k] *
                  A_device[k * (matSize - currentDiagonalIdx) + i];
            }
          });
    });
    Q.wait();

    Q.memcpy(&MA[(currentDiagonalIdx + blockSize) * matSize +
                 (currentDiagonalIdx + blockSize)],
             &A_device[blockSize * (matSize - currentDiagonalIdx - blockSize) +
                       blockSize],
             (matSize - currentDiagonalIdx - blockSize) *
                 (matSize - currentDiagonalIdx - blockSize) * sizeof(T))
        .wait();
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

template <typename T>
bool runExperiment(T* MA, size_t matSize, size_t blockSize,
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
    blockSize = std::stoi(argv[1]);
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
  // std::cout << "***** Original" << std::endl;
  // printMatrix(originalMatrix, matSize);
  // std::cout << "***** Control" << std::endl;
  // printMatrix(controlMatrix, matSize);

  if (gpu) {
    std::cout << "***** GPU" << std::endl;
    memcpy(MA, originalMatrix, matSize * matSize * sizeof(double));

    error = runExperiment(MA, matSize, blockSize, gpu_selector{});
    // printMatrix(MA, matSize);
    if (matSize < 128 && !error)
      error = compareResults(controlMatrix, MA, matSize);

    std::cout << (error ? "Error in computation." : "Success") << std::endl;
  }
  if (cpu) {
    std::cout << "***** CPU" << std::endl;
    memcpy(MA, originalMatrix, matSize * matSize * sizeof(double));

    error = runExperiment(MA, matSize, blockSize, cpu_selector{});
    if (matSize < 128 && !error)
      error = compareResults(controlMatrix, MA, matSize);

    std::cout << (error ? "Error in computation." : "Success") << std::endl;
  }

  delete[] MA;

  return error ? 1 : 0;
}
