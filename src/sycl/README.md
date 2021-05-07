# How to use

## Setup

- Download ComputeCpp CE from [here](https://developer.codeplay.com/products/computecpp/ce/download) and move it to somewhere on your computer (I used `/opt/`)
- Install pre-requisites and drivers you may need (check [here](https://developer.codeplay.com/products/computecpp/ce/guides/))
- Open the file `computecpp.pc` in `${computecpp_path}/lib/pkconfig` and change the value in prefix to the path where you installed ComputeCpp
- Add the `lib/pkconfig` folder to the `PKG_CONFIG_PATH` environment variable by adding the following command to your shell config file (`.bashrc` or `.zshrc`):
```bash
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${computcpp_path}/lib/pkgconfig
```
- Add the library to the `LD_LIBRARY_PATH` environment variable by adding the following command to your shell config file:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${computcpp_path}/lib
```
- Run `sudo ldconfig`

## Build

- Run `make` to compile the file
