#include <CL/sycl.hpp>
#include <iostream>
#include <string>
#include <vector>
using namespace sycl;

int getInputNumber(std::string question, std::string errorMsg, int bottom,
                   int upper) {
  int number;
  const long long MAX_STREAM_SIZE = std::numeric_limits<std::streamsize>::max();

  std::cout << question;

  while (!(std::cin >> number) || number > upper || number < bottom) {
    std::cin.clear();
    if (!std::cin.eof()) {
      std::cin.ignore(MAX_STREAM_SIZE, '\n');
    }

    std::cout << errorMsg << std::endl << std::endl;
    std::cout << question;
  }

  std::cin.ignore(MAX_STREAM_SIZE, '\n');

  return number;
}

cl::sycl::device inputDevice() {
  auto platforms = platform::get_platforms();
  std::vector<cl::sycl::device> devices{};
  int num_devices = 0;

  for (auto const& this_platform : platforms) {
    std::cout << "Platform: " << this_platform.get_info<info::platform::name>()
              << "\n";
    for (auto const& this_device : this_platform.get_devices()) {
      std::cout << "  " << ++num_devices
                << ". Device: " << this_device.get_info<info::device::name>()
                << "\n";
      devices.push_back(this_device);
    }
    std::cout << "\n";
  }

  int choice = getInputNumber(
      "Select device (1-" + std::to_string(devices.size()) + "): ",
      "Invalid device.", 1, devices.size());

  return devices[choice - 1];
}
