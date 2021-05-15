#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>

using namespace cl::sycl;

class matmul_kernel_naive;
class matmul_kernel_blocks;
class matmul_kernel_local_mem;

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

template <typename T>
bool matmulNaive(T* MA, T* MB, T* MC, size_t matSize,
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

  auto device = Q.get_device();

  {
    range<2> dimensions(matSize, matSize);
    const property_list props = {property::buffer::use_host_ptr()};
    buffer<T, 2> bA(MA, dimensions, props);
    buffer<T, 2> bB(MB, dimensions, props);
    buffer<T, 2> bC(MC, dimensions, props);

    Q.submit([&](handler& h) {
      auto a = bA.template get_access<access::mode::read>(h);
      auto b = bB.template get_access<access::mode::read>(h);
      auto c = bC.template get_access<access::mode::write>(h);

      h.parallel_for<matmul_kernel_naive>(range<2>{matSize, matSize},
                                          [=](id<2> idx) {
                                            int j = idx[0];
                                            int i = idx[1];
                                            for (int k = 0; k < matSize; ++k) {
                                              c[j][i] += a[j][k] * b[k][i];
                                            }
                                          });
    });
  }

  Q.wait_and_throw();

  return false;
}

template <typename T>
bool matmulBlocks(T* MA, T* MB, T* MC, size_t matSize,
                  const device_selector& selector) {
  if (!isPowerOfTwo(matSize)) {
    return true;
  }

  queue Q(selector, [&](exception_list eL) {
    try {
      for (auto& e : eL) {
        std::rethrow_exception(e);
      }
    } catch (cl::sycl::exception e) {
      std::cout << " An exception has been thrown: " << e.what() << std::endl;
    }
  });

  auto device = Q.get_device();
  auto maxWorkGroupSize =
      device.get_info<cl::sycl::info::device::max_work_group_size>();
  auto localMemSize = device.get_info<cl::sycl::info::device::local_mem_size>();
  auto blockSize = prevPowerOfTwo(std::sqrt(maxWorkGroupSize));
  std::cout << " The Device Max Work Group Size is : " << maxWorkGroupSize
            << std::endl;
  std::cout << " The Device size of local memory in bytes is : " << localMemSize
            << std::endl;
  std::cout << " The order is : " << matSize << std::endl;
  std::cout << " The blockSize is : " << blockSize << std::endl;

  blockSize = std::min((int)matSize, blockSize);

  {
    range<1> dimensions(matSize * matSize);
    const property_list props = {};
    buffer<T> bA(MA, dimensions, props);
    buffer<T> bB(MB, dimensions, props);
    buffer<T> bC(MC, dimensions, props);

    Q.submit([&](handler& h) {
      auto pA = bA.template get_access<access::mode::read>(h);
      auto pB = bB.template get_access<access::mode::read>(h);
      auto pC = bC.template get_access<access::mode::write>(h);

      h.parallel_for<matmul_kernel_blocks>(
          nd_range<2>{range<2>(matSize, matSize),
                      range<2>(blockSize, blockSize)},
          [=](nd_item<2> item) {
            int blockX = item.get_group(1);
            int blockY = item.get_group(0);

            int localX = item.get_local_id(1);
            int localY = item.get_local_id(0);

            // Start in the A matrix
            int a_start = matSize * blockSize * blockY;
            // End in the A matrix
            int a_end = a_start + matSize - 1;
            // Start in the B matrix
            int b_start = blockSize * blockX;

            // Result for the current C(i,j) element
            T tmp = 0.0f;
            for (int a = a_start, b = b_start; a <= a_end;
                 a += blockSize, b += (blockSize * matSize)) {
              for (int k = 0; k < blockSize; k++) {
                tmp +=
                    pA[a + matSize * localY + k] * pB[b + matSize * localY + k];
              }
            }
            auto elemIndex =
                item.get_global_id(0) * item.get_global_range()[1] +
                item.get_global_id(1);
            // Each thread updates its position
            pC[elemIndex] = tmp;
          });
    });
  }

  Q.wait_and_throw();

  return false;
}

template <typename T>
bool matmulBlocksLocalMem(T* MA, T* MB, T* MC, size_t matSize,
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

  auto device = Q.get_device();
  auto maxWorkGroupSize =
      device.get_info<cl::sycl::info::device::max_work_group_size>();
  auto localMemSize = device.get_info<cl::sycl::info::device::local_mem_size>();
  auto blockSize = prevPowerOfTwo(std::sqrt(maxWorkGroupSize));
  std::cout << " The Device max work group size is : " << maxWorkGroupSize
            << std::endl;
  std::cout << " The Device size of local memory in bytes is : " << localMemSize
            << std::endl;
  std::cout << " The matrixSize is : " << matSize << std::endl;

  if (localMemSize < 2 * blockSize * blockSize * sizeof(T)) {
    blockSize = prevPowerOfTwo(std::sqrt(localMemSize / 2 / sizeof(T)));
  }

  std::cout << " The blockSize is : " << blockSize << std::endl;

  blockSize = std::min((int)matSize, blockSize);

  {
    range<1> dimensions(matSize * matSize);
    const property_list props = {property::buffer::use_host_ptr()};
    buffer<T> bA(MA, dimensions, props);
    buffer<T> bB(MB, dimensions, props);
    buffer<T> bC(MC, dimensions, props);

    Q.submit([&](handler& h) {
      auto pA = bA.template get_access<access::mode::read>(h);
      auto pB = bB.template get_access<access::mode::read>(h);
      auto pC = bC.template get_access<access::mode::write>(h);
      auto localRange = range<1>(blockSize * blockSize);

      accessor<T, 1, access::mode::read_write, access::target::local> pBA(
          localRange, h);
      accessor<T, 1, access::mode::read_write, access::target::local> pBB(
          localRange, h);

      h.parallel_for<matmul_kernel_local_mem>(
          nd_range<2>{range<2>(matSize, matSize),
                      range<2>(blockSize, blockSize)},
          [=](nd_item<2> item) {
            int blockX = item.get_group(1);
            int blockY = item.get_group(0);
            int localX = item.get_local_id(1);
            int localY = item.get_local_id(0);

            // Start index for A matrix
            int a_start = matSize * blockSize * blockY;
            // End index for A matrix
            int a_end = a_start + matSize - 1;
            // Start index for B matrix
            int b_start = blockSize * blockX;

            // Result for the current C(i,j) element
            T tmp = 0.0f;

            for (int a = a_start, b = b_start; a <= a_end;
                 a += blockSize, b += (blockSize * matSize)) {
              // Coolaborative loading of blocks into shared memory
              pBA[localY * blockSize + localX] =
                  pA[a + matSize * localY + localX];
              pBB[localX * blockSize + localY] =
                  pB[b + matSize * localY + localX];

              item.barrier(access::fence_space::local_space);

              for (int k = 0; k < blockSize; k++) {
                tmp +=
                    pBA[localY * blockSize + k] * pBB[localX * blockSize + k];
              }
              item.barrier(access::fence_space::local_space);
            }

            auto elemIndex =
                item.get_global_id(0) * item.get_global_range()[1] +
                item.get_global_id(1);

            pC[elemIndex] = tmp;
          });
    });
  }

  Q.wait_and_throw();

  return false;
}

void usage(std::string programName) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: " << std::endl;
  std::cout << programName << " [matrix size] [omp|sycl]" << std::endl;
  std::cout << "[matrix size] : Size of the matrix to multiply (minimum 32)"
            << std::endl;
  std::cout << "[gpu|cpu]    : Use GPU or CPU device."
            << " Default is to use both " << std::endl;
}

template <typename T>
bool runExperimentNaive(T* MA, T* MB, T* MC, size_t matSize,
                        const device_selector& selector) {
  auto start = std::chrono::steady_clock::now();
  bool error = matmulNaive(MA, MB, MC, matSize, selector);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();

  std::cout << "Time: " << time << std::endl;
  float flops =
      (2.0f * matSize * matSize * matSize / (time / 1000.0f)) * 1.0e-9f;
  std::cout << "GFLOPs: " << flops << std::endl;

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

  return error;
}

template <typename T>
bool runExperimentBlocks(T* MA, T* MB, T* MC, size_t matSize,
                         const device_selector& selector) {
  auto start = std::chrono::steady_clock::now();
  bool error = matmulBlocks(MA, MB, MC, matSize, selector);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();

  std::cout << "Time: " << time << std::endl;
  float flops =
      (2.0f * matSize * matSize * matSize / (time / 1000.0f)) * 1.0e-9f;
  std::cout << "GFLOPs: " << flops << std::endl;

  if (!error) {
    error = false;
    // Testing
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        if (std::fabs(MC[i * matSize + j] - MB[i * matSize + j]) > 1e-8) {
          // std::cout << " Position " << i << ", " << j
          //           << " differs: " << MC[i * matSize + j]
          //           << " != " << MB[i * matSize + j] << std::endl;
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

  return error;
}

template <typename T>
bool runExperimentBlocksLocalMem(T* MA, T* MB, T* MC, size_t matSize,
                                 const device_selector& selector) {
  auto start = std::chrono::steady_clock::now();
  bool error = matmulBlocksLocalMem(MA, MB, MC, matSize, selector);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();

  std::cout << "Time: " << time << std::endl;
  float flops =
      (2.0f * matSize * matSize * matSize / (time / 1000.0f)) * 1.0e-9f;
  std::cout << "GFLOPs: " << flops << std::endl;

  if (!error) {
    error = false;
    // Testing
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        if (std::fabs(MC[i * matSize + j] - MB[i * matSize + j]) > 1e-8) {
          // std::cout << " Position " << i << ", " << j
          //           << " differs: " << MC[i * matSize + j]
          //           << " != " << MB[i * matSize + j] << std::endl;
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

  return error;
}

int main(int argc, char* argv[]) {
  bool gpu = true;
  bool cpu = true;
  bool error = false;

  if (argc != 2 && argc != 3) {
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

  if (matSize < 32) {
    usage(argv[0]);
    return 1;
  }

  if (argc == 3) {
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

  if (gpu) {
    std::cout << "***** GPU - Blocks with local memory " << std::endl;
    // Matrix initialization
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        MC[i * matSize + j] = 0.0f;  // i * matSize + j;
      }

    error = runExperimentBlocksLocalMem(MA, MB, MC, matSize, gpu_selector{});
  }
  if (cpu) {
    std::cout << "***** CPU - Blocks with local memory" << std::endl;
    // Matrix initialization
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        MC[i * matSize + j] = 0.0f;  // i * matSize + j;
      }

    error = runExperimentBlocksLocalMem(MA, MB, MC, matSize, cpu_selector{});
  }

  // if (gpu) {
  //   std::cout << "***** Naive GPU " << std::endl;
  //   // Matrix initialization
  //   for (int i = 0; i < matSize; i++)
  //     for (int j = 0; j < matSize; j++) {
  //       MC[i * matSize + j] = 0.0f;  // i * matSize + j;
  //     }

  //   error = runExperimentNaive(MA, MB, MC, matSize,
  //   gpu_selector{});
  // }
  // if (cpu) {
  //   std::cout << "***** Naive CPU " << std::endl;
  //   // Matrix initialization
  //   for (int i = 0; i < matSize; i++)
  //     for (int j = 0; j < matSize; j++) {
  //       MC[i * matSize + j] = 0.0f;  // i * matSize + j;
  //     }

  //   error = runExperimentNaive(MA, MB, MC, matSize,
  //   cpu_selector{});
  // }

  if (gpu) {
    std::cout << "***** GPU - Blocks " << std::endl;
    // Matrix initialization
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        MC[i * matSize + j] = 0.0f;  // i * matSize + j;
      }

    error = runExperimentBlocks(MA, MB, MC, matSize, gpu_selector{});
  }
  if (cpu) {
    std::cout << "***** CPU - Blocks " << std::endl;
    // Matrix initialization
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        MC[i * matSize + j] = 0.0f;  // i * matSize + j;
      }

    error = runExperimentBlocks(MA, MB, MC, matSize, cpu_selector{});
  }

  delete[] MA;
  delete[] MB;
  delete[] MC;

  return error ? 1 : 0;
}
