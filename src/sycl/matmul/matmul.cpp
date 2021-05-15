#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>

#include "helper.h"
#include "mm_blocks.h"
#include "mm_local_mem.h"
#include "mm_naive.h"

using namespace cl::sycl;

template <typename T>
bool runExperiments(T* MA, T* MB, T* MC, size_t matSize, int op,
                    const device_selector& selector) {
  auto start = std::chrono::steady_clock::now();
  bool error;
  switch (op) {
    case 1:
      error = matmulNaive(MA, MB, MC, matSize, selector);
      break;
    case 2:
      error = matmulBlocks(MA, MB, MC, matSize, selector);
      break;
    case 3:
      error = matmulBlocksLocalMem(MA, MB, MC, matSize, selector);
      break;
    default:
      error = true;
      break;
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
  float flops =
      (2.0f * matSize * matSize * matSize / (time / 1000.0f)) * 1.0e-9f;

  std::cout << "Time: " << time << std::endl;
  std::cout << "GFLOPs: " << flops << std::endl;

  testError(error, matSize, MB, MC);

  return error;
}

int main(int argc, char* argv[]) {
  bool gpu = true;
  bool cpu = true;
  bool error = false;
  size_t matSize = 0;
  size_t blockSize = 0;
  int op = 0;

  if (argc != 4 && argc != 5) {
    usage(argv[0]);
    return 1;
  }

  try {
    matSize = std::stoi(argv[1]);
  } catch (...) {
    usage(argv[0]);
    return 1;
  }

  if (std::string(argv[2]) == "gpu") {
    gpu = true;
    cpu = false;
  } else if (std::string(argv[2]) == "cpu") {
    gpu = false;
    cpu = true;
  } else {
    usage(argv[0]);
    return 1;
  }
  
  try {
    op = std::stoi(argv[3]);

    if (op <= 0 || op >= 4){
      usage(argv[0]);
      return 1;
    }
  } catch (...) {
    usage(argv[0]);
    return 1;
  }

  if(argc == 5) {
    try {
      blockSize = std::stoi(argv[4]);
    } catch (...) {
      usage(argv[0]);
      return 1;
    }
  }

  double* MA = new double[matSize * matSize];
  double* MB = new double[matSize * matSize];
  double* MC = new double[matSize * matSize];

  // Matrix initialization
  for (int i = 0; i < matSize; i++)
    for (int j = 0; j < matSize; j++) {
      MA[i * matSize + j] = 0.0f;
      if (i == j) {
        MA[i * matSize + j] = 1.0f;
      }
      MB[i * matSize + j] = 2.0f;
      MC[i * matSize + j] = 0.0f;
    }

  for (int i = 0; i < matSize; i++)
    for (int j = 0; j < matSize; j++) {
      MC[i * matSize + j] = 0.0f;  // i * matSize + j;
    }

  if(gpu)
    runExperiments(MA, MB, MC, matSize, op, gpu_selector{});
  else
    runExperiments(MA, MB, MC, matSize, op, cpu_selector{});

  delete[] MA;
  delete[] MB;
  delete[] MC;

  return error ? 1 : 0;
}
