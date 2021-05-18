#include <iostream>
#include <iomanip>
#include <CL/sycl.hpp>

using namespace cl::sycl;
using namespace std;

void printMatrix(double* matrix, size_t size) {
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      std::cout << std::setw(12) << matrix[i * size + j];
    }
    std::cout << std::endl;
  }
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


int getRandBetween(int min, int max) { return rand() % (max - min + 1) + min; }

void outputDevInfo(const sycl::device& dev) {
  std::cout << "  -> Selected device: "
            << dev.get_info<sycl::info::device::name>() << std::endl;
  std::cout << "  -> Device vendor: "
            << dev.get_info<sycl::info::device::vendor>() << std::endl;
}

void usage(std::string programName) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: " << std::endl;
  std::cout << programName << " [matrix size] [gpu|cpu] [op]" << std::endl;
  std::cout << "[matrix size] : Size of the matrix to multiply"  << std::endl;
  std::cout << "[block size]  : Size of blocks to use" << std::endl;
  std::cout << "[op]          : Algorithm to use" << std::endl;
  std::cout << "[gpu|cpu]     : Use GPU or CPU device." << std::endl;
  std::cout << "[runs]        : Number of runs to execute" << std::endl;
}
