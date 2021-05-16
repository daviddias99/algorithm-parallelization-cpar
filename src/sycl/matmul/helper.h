#ifndef HELPER_H
#define HELPER_H

#include <CL/sycl.hpp>
#include <iostream>

#define TEST_MODE false

using namespace cl::sycl;

void outputDevInfo(const sycl::device& dev) {
  std::cout << "  -> Selected device: "
            << dev.get_info<sycl::info::device::name>() << std::endl;
  std::cout << "  -> Device vendor: "
            << dev.get_info<sycl::info::device::vendor>() << std::endl;
}

inline int prevPowerOfTwo(int x) {
  if (x < 0) {
    return 0;
  }
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x - (x >> 1);
}

inline bool isPowerOfTwo(int x) { return (x & (x - 1)) == 0; }

void usage(std::string programName) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: " << std::endl;
  std::cout << programName << " [matrix size] [gpu|cpu] [op]" << std::endl;
  std::cout << "[matrix size] : Size of the matrix to multiply"  << std::endl;
  std::cout << "[block size]  : Size of blocks to use" << std::endl;
  std::cout << "[op]          : Algorithm to use (1. naive, 2. block, 3. block + local mem.)" << std::endl;
  std::cout << "[gpu|cpu]     : Use GPU or CPU device." << std::endl;
  std::cout << "[runs]        : Number of runs to execute" << std::endl;
}

template <typename T>
void testError(bool& error, size_t matSize, T* MB, T* MC) {
  if (!error) {
    error = false;
    // Testing
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        if (std::fabs(MC[i * matSize + j] - MB[i * matSize + j]) > 1e-8) {
          std::cout << " Position " << i << ", " << j
                    << " differs: " << MC[i * matSize + j]
                    << " != " << MB[i * matSize + j] << std::endl;
          error = true;
        }
      }
    if (!error) {
      std::cout << "Success" << std::endl;
      ;
    } else {
      std::cout << "Error in the computation " << std::endl;
    }
  }
}

#endif