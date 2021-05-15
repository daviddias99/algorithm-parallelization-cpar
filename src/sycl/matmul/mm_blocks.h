#include <CL/sycl.hpp>
#include <cmath>
#include <iostream>
#include "helper.h"

class matmul_kernel_blocks;
using namespace cl::sycl;

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
