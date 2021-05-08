#include <CL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>

using namespace cl::sycl;

class mxm_kernel;
class mxm_kernel_naive;

void outputDevInfo(const sycl::device& dev) {
  std::cout << "  -> Selected device: "
            << dev.get_info<sycl::info::device::name>() << std::endl;
  std::cout << "  -> Device vendor: "
            << dev.get_info<sycl::info::device::vendor>() << std::endl;
}

/* Obtains the previous power of two from the given integer.
 * It works by masking out all ones after the first one bit,
 * then leaves the first one bit intact, effectively
 * yielding the first power of two < x. */
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

/* Checks if X is a power of two.
 * If there are bits sets to one after AND with the
 * previous number, then it is not a power of two.
 */
inline bool isPowerOfTwo(int x) { return (x & (x - 1)) == 0; }

template <typename T>
bool local_mxm(T* MA, T* MB, T* MC, size_t matSize,
               const device_selector& selector) {
  // Make sure it is power of two before running
  if (!isPowerOfTwo(matSize)) {
    std::cout << " This example only works with power of two sizes "
              << std::endl;
    return true;
  }

  queue q(selector, [&](exception_list eL) {
    try {
      for (auto& e : eL) {
        std::rethrow_exception(e);
      }
    } catch (cl::sycl::exception e) {
      std::cout << " An exception has been thrown: " << e.what() << std::endl;
    }
  });

  auto device = q.get_device();
  auto maxBlockSize =
      device.get_info<cl::sycl::info::device::max_work_group_size>();
  auto localMemSize = device.get_info<cl::sycl::info::device::local_mem_size>();
  auto blockSize = prevPowerOfTwo(std::sqrt(maxBlockSize));
  std::cout << " The Device Max Work Group Size is : " << maxBlockSize
            << std::endl;
  std::cout << " The Device size of local memory in bytes is : " << localMemSize
            << std::endl;
  std::cout << " The order is : " << matSize << std::endl;
  std::cout << " The blockSize is : " << blockSize << std::endl;
  // Make sure the block size is not larger than the mat size
  blockSize = std::min((int) matSize, blockSize);

  {
    /* Buffers can be constructed with property lists. In this example,
     * the buffer is given the property "use host pointer", which tells
     * the runtime to use the host pointer for all data storage (instead
     * of making copies internally). Additionally, when running on a
     * device that shares memory with the host (for example a CPU),
     * "zero-copy" memory optimisations can be used by the driver. */
    range<1> dimensions(matSize * matSize);
    const property_list props = {property::buffer::use_host_ptr()};
    buffer<T> bA(MA, dimensions, props);
    buffer<T> bB(MB, dimensions, props);
    buffer<T> bC(MC, dimensions, props);

    q.submit([&](handler& cgh) {
      auto pA = bA.template get_access<access::mode::read>(cgh);
      auto pB = bB.template get_access<access::mode::read>(cgh);
      auto pC = bC.template get_access<access::mode::write>(cgh);
      auto localRange = range<1>(blockSize * blockSize);

      accessor<T, 1, access::mode::read_write, access::target::local> pBA(
          localRange, cgh);
      accessor<T, 1, access::mode::read_write, access::target::local> pBB(
          localRange, cgh);

      cgh.parallel_for<mxm_kernel>(
          nd_range<2>{range<2>(matSize, matSize),
                      range<2>(blockSize, blockSize)},
          [=](nd_item<2> item) {
            // Current block
            int blockX = item.get_group(1);
            int blockY = item.get_group(0);

            // Current local item
            int localX = item.get_local_id(1);
            int localY = item.get_local_id(0);

            // Start in the A matrix
            int a_start = matSize * blockSize * blockY;
            // End in the b matrix
            int a_end = a_start + matSize - 1;
            // Start in the b matrix
            int b_start = blockSize * blockX;

            // Result for the current C(i,j) element
            T tmp = 0.0f;
            // We go through all a, b blocks
            for (int a = a_start, b = b_start; a <= a_end;
                 a += blockSize, b += (blockSize * matSize)) {
              // Copy the values in shared memory collectively
              pBA[localY * blockSize + localX] =
                  pA[a + matSize * localY + localX];
              // Note the swap of X/Y to maintain contiguous access
              pBB[localX * blockSize + localY] =
                  pB[b + matSize * localY + localX];
              item.barrier(access::fence_space::local_space);
              // Now each thread adds the value of its sum
              for (int k = 0; k < blockSize; k++) {
                tmp +=
                    pBA[localY * blockSize + k] * pBB[localX * blockSize + k];
              }
              // The barrier ensures that all threads have written to local
              // memory before continuing
              item.barrier(access::fence_space::local_space);
            }
            auto elemIndex =
                item.get_global_id(0) * item.get_global_range()[1] +
                item.get_global_id(1);
            // Each thread updates its position
            pC[elemIndex] = tmp;
          });
    });
  }

  q.wait_and_throw();

  return false;
}

template <typename T>
bool local_mxm_naive(T* MA, T* MB, T* MC, size_t matSize,
                     const device_selector& selector) {
  // Make sure it is power of two before running
  if (!isPowerOfTwo(matSize)) {
    std::cout << " This example only works with power of two sizes "
              << std::endl;
    return true;
  }

  queue q(selector, [&](exception_list eL) {
    try {
      for (auto& e : eL) {
        std::rethrow_exception(e);
      }
    } catch (cl::sycl::exception e) {
      std::cout << " An exception has been thrown: " << e.what() << std::endl;
    }
  });

  auto device = q.get_device();

  {
    range<2> dimensions(matSize, matSize);
    const property_list props = {property::buffer::use_host_ptr()};
    buffer<T, 2> bA(MA, dimensions, props);
    buffer<T, 2> bB(MB, dimensions, props);
    buffer<T, 2> bC(MC, dimensions, props);

    q.submit([&](handler& h) {
      auto a = bA.template get_access<access::mode::read>(h);
      auto b = bB.template get_access<access::mode::read>(h);
      auto c = bC.template get_access<access::mode::write>(h);

      h.parallel_for<mxm_kernel_naive>(range<2>{matSize, matSize}, [=](id<2> idx) {
        int j = idx[0];
        int i = idx[1];
        for (int k = 0; k < matSize; ++k) {
          c[j][i] += a[j][k] * b[k][i];
        }
      });
    });
  }

  q.wait_and_throw();

  return false;
}

/* Helper function to indicate the parameters the sample takes. */
void usage(std::string programName) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: " << std::endl;
  std::cout << programName << " [matrix size] [omp|sycl]" << std::endl;
  std::cout << "[matrix size] : Size of the matrix to multiply (minimum 32)"
            << std::endl;
  std::cout << "[gpu|cpu]    : Use GPU or CPU device."
            << " Default is to use both " << std::endl;
}

bool runExperiment(float* MA, float* MB, float* MC, size_t matSize,
                   const device_selector& selector) {
  auto start = std::chrono::steady_clock::now();
  bool error = local_mxm(MA, MB, MC, matSize, selector);
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

bool runExperimentNaive(float* MA, float* MB, float* MC, size_t matSize,
                        const device_selector& selector) {
  auto start = std::chrono::steady_clock::now();
  bool error = local_mxm_naive(MA, MB, MC, matSize, selector);
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

int main(int argc, char* argv[]) {
  float* MA;
  float* MB;
  float* MC;
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
    }
  }

  MA = new float[matSize * matSize];
  MB = new float[matSize * matSize];
  MC = new float[matSize * matSize];

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
    std::cout << "***** GPU " << std::endl;
    // Matrix initialization
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        MC[i * matSize + j] = 0.0f;  // i * matSize + j;
      }

    error = runExperiment(MA, MB, MC, matSize, gpu_selector{});
  }
  if (cpu) {
    std::cout << "***** CPU " << std::endl;
    // Matrix initialization
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        MC[i * matSize + j] = 0.0f;  // i * matSize + j;
      }

    error = runExperiment(MA, MB, MC, matSize, cpu_selector{});
  }

  if (gpu) {
    std::cout << "***** Naive GPU " << std::endl;
    // Matrix initialization
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        MC[i * matSize + j] = 0.0f;  // i * matSize + j;
      }

    error = runExperimentNaive(MA, MB, MC, matSize, gpu_selector{});
  }
  if (cpu) {
    std::cout << "***** Naive CPU " << std::endl;
    // Matrix initialization
    for (int i = 0; i < matSize; i++)
      for (int j = 0; j < matSize; j++) {
        MC[i * matSize + j] = 0.0f;  // i * matSize + j;
      }

    error = runExperimentNaive(MA, MB, MC, matSize, cpu_selector{});
  }

  delete[] MA;
  delete[] MB;
  delete[] MC;

  return error ? 1 : 0;
}
