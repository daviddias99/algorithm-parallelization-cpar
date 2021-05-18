#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#include "../../omp/lu/lu_blocks_common.h"
#include "../../omp/lu/lu_seq.h"
#include "helper.h"

#define TEST_MODE true

using namespace cl::sycl;
using namespace std;

class lu_kernel;
class lu_kernel_2;
class lu_kernel_3;
class lu_kernel_0;

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

  double* diagonalBlock = MA;

  for (size_t currentDiagonalIdx = 0; currentDiagonalIdx < matSize;
       currentDiagonalIdx += blockSize) {
    luSequential(diagonalBlock, blockSize, blockSize, matSize);

    if (matSize - currentDiagonalIdx <= blockSize) break;

    range<2> dimensions((matSize), (matSize));
    const property_list props = {};
    buffer<double, 2> matrix(MA, dimensions, props);

    // Q.submit([&](handler& h) {
    //   auto matrixAcc = matrix.template
    //   get_access<access::mode::read_write>(h);

    //   h.single_task<lu_kernel_0>([=]() {
    //     for (size_t k = currentDiagonalIdx;
    //          k < currentDiagonalIdx + blockSize && matrixAcc[k][k] != 0; k++)
    //          {
    //       for (size_t i = k + 1; i < blockSize; i++) {
    //         matrixAcc[i][k] /= matrixAcc[k][k];
    //       }

    //       for (size_t i = k + 1; i < blockSize; i++) {
    //         for (size_t j = k + 1; j < blockSize; j++) {
    //           matrixAcc[i][j] -= matrixAcc[i][k] * matrixAcc[k][j];
    //         }
    //       }
    //     }
    //   });
    // });
    // Q.wait_and_throw();

    if (matSize - currentDiagonalIdx <= blockSize) break;

    size_t nBlocks = (matSize - currentDiagonalIdx - blockSize) / blockSize;

    Q.submit([&](handler& h) {
      auto matrixAcc = matrix.template get_access<access::mode::read_write>(h);

      h.parallel_for<lu_kernel_3>(range<1>{range<1>(nBlocks)}, [=](id<1> id) {
        int jj = id.get(0) * blockSize + blockSize + currentDiagonalIdx;

        for (size_t k = currentDiagonalIdx; k < currentDiagonalIdx + blockSize;
             k++) {
          for (size_t i = k + 1; i < currentDiagonalIdx + blockSize; i++) {
            for (size_t j = jj; j < jj + blockSize; j++) {
              matrixAcc[i][j] -= matrixAcc[i][k] * matrixAcc[k][j];
            }
          }
        }
      });
    });

    Q.submit([&](handler& h) {
      auto matrixAcc = matrix.template get_access<access::mode::read_write>(h);

      h.parallel_for<lu_kernel_2>(range<1>{range<1>(nBlocks)}, [=](id<1> id) {
        int ii = id.get(0) * blockSize + blockSize + currentDiagonalIdx;

        for (size_t k = currentDiagonalIdx; k < currentDiagonalIdx + blockSize;
             k++) {
          for (size_t i = ii; i < ii + blockSize; i++) {
            matrixAcc[i][k] /= matrixAcc[k][k];
          }
          for (size_t i = ii; i < ii + blockSize; i++) {
            for (size_t j = k + 1; j < currentDiagonalIdx + blockSize; j++) {
              matrixAcc[i][j] -= matrixAcc[i][k] * matrixAcc[k][j];
            }
          }
        }
      });
    });
    Q.wait_and_throw();

    size_t subMatrixSize = matSize - (blockSize + currentDiagonalIdx);

    Q.submit([&](handler& h) {
      auto matrixAcc = matrix.template get_access<access::mode::read_write>(h);

      h.parallel_for<lu_kernel>(
          range<2>{range<2>(subMatrixSize, subMatrixSize)}, [=](id<2> id) {
            int j = id.get(1) + blockSize + currentDiagonalIdx;
            int i = id.get(0) + blockSize + currentDiagonalIdx;
            double tmp = 0.0;

            for (int k = currentDiagonalIdx; k < currentDiagonalIdx + blockSize;
                 k++) {
              tmp += matrixAcc[i][k] * matrixAcc[k][j];
            }
            matrixAcc[i][j] -= tmp;
          });
    });
    Q.wait_and_throw();

    diagonalBlock += blockSize * matSize + blockSize;
  }

  return false;
}

bool runExperiments(double* MA, size_t matSize, size_t blockSize, int op,
                    const device_selector& selector, int numExperiments,
                    double* controlMatrix) {
  bool error;

  for (size_t i = 0; i < numExperiments; i++) {
    auto start = std::chrono::steady_clock::now();
    switch (op) {
      case 1:
        error = luFactorization(MA, matSize, blockSize, selector);
        break;
      default:
        error = true;
        break;
    }

    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    if (TEST_MODE) {
      float flops =
          (2.0f / 3 * matSize * matSize * matSize / (time / 1000000.0f)) *
          1.0e-9f;
      std::cout << "Time: " << time << std::endl;
      std::cout << "GFLOPs: " << flops << std::endl;
      if (matSize < 128 && !error) {
        error = compareResults(controlMatrix, MA, matSize);
      }
    } else {
      std::cout << op << " " << matSize << " " << blockSize << " "
                << time / 1000000.0 << " N/A" << std::endl;
    }
  }
  return error;
}

int main(int argc, char* argv[]) {
  bool gpu = true;
  bool cpu = true;
  bool error = false;
  size_t matSize = 0;
  size_t blockSize = 0;
  int op = 0;
  int nruns = 1;
  if (argc != 5 && argc != 6) {
    usage(argv[0]);
    return 1;
  }

  try {
    matSize = std::stoi(argv[1]);
  } catch (...) {
    usage(argv[0]);
    return 1;
  }

  try {
    blockSize = std::stoi(argv[2]);
  } catch (...) {
    usage(argv[0]);
    return 1;
  }

  try {
    op = std::stoi(argv[3]);

    if (op <= 0 || op >= 4) {
      usage(argv[0]);
      return 1;
    }
  } catch (...) {
    usage(argv[0]);
    return 1;
  }

  if (std::string(argv[4]) == "gpu") {
    gpu = true;
    cpu = false;
  } else if (std::string(argv[4]) == "cpu") {
    gpu = false;
    cpu = true;
  } else {
    usage(argv[0]);
    return 1;
  }

  if (argc == 6) {
    try {
      nruns = std::stoi(argv[5]);
    } catch (...) {
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

  if (TEST_MODE) {
    memcpy(controlMatrix, originalMatrix, matSize * matSize * sizeof(double));
    luSequential(controlMatrix, matSize, matSize, matSize);
  }

  if (gpu)
    error = runExperiments(originalMatrix, matSize, blockSize, op,
                           gpu_selector{}, nruns, controlMatrix);
  else
    error = runExperiments(originalMatrix, matSize, blockSize, op,
                           cpu_selector{}, nruns, controlMatrix);

  std::cout << (error ? "Error in computation." : "Success") << std::endl;

  delete[] originalMatrix;
  delete[] controlMatrix;
  delete[] MA;

  return error ? 1 : 0;
}
