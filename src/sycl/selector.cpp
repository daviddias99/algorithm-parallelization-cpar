#include <CL/sycl.hpp>
#include <iostream>
#include <string>

void output_dev_info(const sycl::device& dev,
                     const std::string& selector_name) {
  std::cout << selector_name << ": " << std::endl
            << "  -> Selected device: " << dev.get_info<sycl::info::device::name>() << std::endl;
  std::cout << "  -> Device vendor: "
            << dev.get_info<sycl::info::device::vendor>() << std::endl;
}

int main() {
  output_dev_info(sycl::device{sycl::cpu_selector{}}, "cpu_selector");
  output_dev_info(sycl::device{sycl::gpu_selector{}}, "gpu_selector");
}