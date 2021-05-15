#include <CL/sycl.hpp>
#include <cmath>
#include <iostream>
#include "helper.h"


using namespace cl::sycl;
class matmul_kernel_naive;

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
