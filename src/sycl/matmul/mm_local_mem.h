#include <CL/sycl.hpp>
#include <cmath>
#include <iostream>

#include "helper.h"

using namespace cl::sycl;
class matmul_kernel_local_mem;

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
