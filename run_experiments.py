from datetime import datetime
import os
import subprocess
import pandas as pd
from datetime import datetime
import json

os.makedirs('out', exist_ok=True)
os.makedirs('results', exist_ok=True)

dir = os.path.dirname(__file__)
src_path = os.path.join(dir, 'src')
out_path = os.path.join(dir, 'out')
results_path = os.path.join(dir, 'results')

def compile_cuda(alg, file):
  subprocess.run(['nvcc', os.path.join(src_path, 'cuda', alg, file), '-o', os.path.join(out_path, f'{alg}_cuda')])

def compile_omp(alg, file):
  subprocess.run(['g++', '-O2', os.path.join(src_path, 'omp', alg, file), '-o', os.path.join(out_path, f'{alg}_omp'), '-fopenmp'])

def compile_sycl_gpu(alg, file):
  subprocess.run(['clang++', '-fsycl', '-fsycl-targets=nvptx64-nvidia-cuda-sycldevice', os.path.join(src_path, 'sycl', alg, file), '-o', os.path.join(out_path, f'{alg}_sycl_gpu')]) 

def compile_sycl_cpu(alg, file):
  subprocess.run(['make', file], cwd= os.path.join(src_path, 'sycl', alg)) 
  subprocess.run(['mv', os.path.join(src_path, 'sycl', alg, file), os.path.join(out_path, f'{alg}_sycl_cpu')]) 

def run(file, size, *args):
  cmd = [os.path.join(out_path, file), size, *args]
  process = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)
  return process.stdout

def parse_output(output, other_data):
    return [other_data + row.split(',') for row in output.split('\n') if row.strip() != '']


experiments_lu = [
  ['lu_omp', 1, 3, 1],
  ['lu_omp', 2, 3, 128],
  ['lu_omp', 2, 3, 256],
  ['lu_omp', 2, 3, 512],
  ['lu_omp', 3, 3, 128],
  ['lu_omp', 3, 3, 256],
  ['lu_omp', 3, 3, 512],
  ['lu_omp', 4, 3, 128],
  ['lu_omp', 4, 3, 256],
  ['lu_omp', 4, 3, 512],
  ['lu_sycl_cpu', 8 , 1, 'cpu', 3],
  ['lu_sycl_cpu', 16, 1, 'cpu', 3],
  ['lu_sycl_cpu', 32, 1, 'cpu', 3],
  ['lu_sycl_gpu', 8 , 1, 'gpu', 3],
  ['lu_sycl_gpu', 16, 1, 'gpu', 3],
  ['lu_sycl_gpu', 32, 1, 'gpu', 3],
]

experiments_mm = [
  ['matmul_omp', 1, 3, 1],
  ['matmul_omp', 3, 3, 128],
  ['matmul_omp', 3, 3, 256],
  ['matmul_omp', 3, 3, 512],
  ['matmul_cuda', 2, 3, 8],
  ['matmul_cuda', 2, 3, 16],
  ['matmul_cuda', 2, 3, 32],
  ['matmul_cuda', 3, 3, 8],
  ['matmul_cuda', 3, 3, 16],
  ['matmul_cuda', 3, 3, 32],
  ['matmul_sycl_cpu', 8 , 1, 'cpu', 3],
  ['matmul_sycl_cpu', 16, 1, 'cpu', 3],
  ['matmul_sycl_cpu', 32, 1, 'cpu', 3],
  ['matmul_sycl_cpu', 8 , 2, 'cpu', 3],
  ['matmul_sycl_cpu', 16, 2, 'cpu', 3],
  ['matmul_sycl_cpu', 32, 2, 'cpu', 3],
  ['matmul_sycl_cpu', 8 , 3, 'cpu', 3],
  ['matmul_sycl_cpu', 16, 3, 'cpu', 3],
  ['matmul_sycl_cpu', 32, 3, 'cpu', 3],
  ['matmul_sycl_gpu', 8 , 1, 'gpu', 3],
  ['matmul_sycl_gpu', 16, 1, 'gpu', 3],
  ['matmul_sycl_gpu', 32, 1, 'gpu', 3],
  ['matmul_sycl_gpu', 8 , 2, 'gpu', 3],
  ['matmul_sycl_gpu', 16, 2, 'gpu', 3],
  ['matmul_sycl_gpu', 32, 2, 'gpu', 3],
  ['matmul_sycl_gpu', 8 , 3, 'gpu', 3],
  ['matmul_sycl_gpu', 16, 3, 'gpu', 3],
  ['matmul_sycl_gpu', 32, 3, 'gpu', 3],
]

def run_experiment(exp):

  results = []

  for size in range(1024, 8192+1024, 1024):
    print('\t\tMatrix size {} ...'.format(size))

    output = run(exp[0],str(size), *list(map(lambda x: str(x), exp[1:])))
    parsed_output = parse_output(output, "")
    results += parsed_output

  return pd.DataFrame(results, columns=['Language', 'Algorithm', 'Matrix Size', 'Time', 'L1 DCM', 'L2 DCM'])


compile_cuda('matmul', 'matmul.cu')
compile_omp('lu_decomp', 'lu.cpp')
compile_omp('matmul', 'matmul.cpp')
compile_sycl_gpu('lu_decomp', 'lu.cpp')
compile_sycl_gpu('matmul', 'matmul.cpp')
compile_sycl_cpu('lu_decomp', 'lu')
compile_sycl_cpu('matmul', 'matmul')
