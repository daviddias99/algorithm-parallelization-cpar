from datetime import datetime
import os
import subprocess
import pandas as pd
from datetime import datetime
import exps
from tqdm import tqdm

os.makedirs('out', exist_ok=True)
os.makedirs('results', exist_ok=True)

dir = os.path.dirname(__file__)
src_path = os.path.join(dir, 'src')
out_path = os.path.join(dir, 'out')
results_path = os.path.join(dir, 'results')

def compile_cuda(alg, file):
  subprocess.run([f'nvcc', os.path.join(src_path, 'cuda', alg, file), '-o', os.path.join(out_path, f'{alg}_cuda')])

def compile_omp(alg, file, pfile, pvalue):

  for i in range(pvalue):
    subprocess.run(f"sed 's/\(#define NUM_THREADS \).*/\\1{i+1}/' {os.path.join(src_path, 'omp', alg, pfile)} -i", shell=True)
    subprocess.run(['g++', '-O2', file, '-o', os.path.join(out_path, f'{alg}_omp_{i+1}'), '-fopenmp'], cwd=os.path.join(src_path, 'omp', alg))

def compile_sycl_gpu(alg, file):
  subprocess.run(['clang++', '-fsycl', '-fsycl-targets=nvptx64-nvidia-cuda-sycldevice', file, '-o', os.path.join(out_path, f'{alg}_sycl_gpu')], cwd=os.path.join(src_path, 'sycl', alg))

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
    return [other_data + row.split(' ') for row in output.split('\n') if row.strip() != '']

LOWER_BOUND = 1024
UPPER_BOUND = 8192
STEP = 1024

def run_experiment(exp):

  results = []

  for size in tqdm(range(LOWER_BOUND, UPPER_BOUND+STEP, STEP), f'{exp[0]} - Matrix Size'):
    # print('\t\tMatrix size {} ...'.format(size))

    output = run(exp[0],str(size), *list(map(lambda x: str(x), exp[1:])))
    parsed_output = parse_output(output, [exp[0]])
    results += parsed_output

  return pd.DataFrame(results, columns=['Exp', 'Op', 'Matrix Size', 'Block Size', 'Time', 'P'])


compile_cuda('matmul', 'matmul.cu')
compile_omp('lu', 'lu.cpp', 'lu_seq.h', 4)
compile_omp('matmul', 'matmul.cpp', 'matmul.cpp',4)
compile_sycl_gpu('lu', 'lu.cpp')
compile_sycl_gpu('matmul', 'matmul.cpp')
compile_sycl_cpu('lu', 'lu')
compile_sycl_cpu('matmul', 'matmul')

print('LU Sequential')
for exp in tqdm(exps.experiments_lu_seq):
  res_path = os.path.join(results_path, 'exp_{}_{}.csv'.format('lu_seq', datetime.now()))
  res = run_experiment(exp)
  res.to_csv(res_path, index=False, header=True)

print('LU Data Parallel')
for exp in tqdm(exps.experiments_lu_par_data):
  res_path = os.path.join(results_path, 'exp_{}_{}.csv'.format('lu_data', datetime.now()))
  res = run_experiment(exp)
  res.to_csv(res_path, index=False, header=True)

print('LU Func Parallel')
for exp in tqdm(exps.experiments_lu_par_func):
  res_path = os.path.join(results_path, 'exp_{}_{}.csv'.format('lu_func', datetime.now()))
  res = run_experiment(exp)
  res.to_csv(res_path, index=False, header=True)

print('LU Sycl CPU')
for exp in tqdm(exps.experiments_lu_sycl_cpu):
  res_path = os.path.join(results_path, 'exp_{}_{}.csv'.format('lu_sycl_cpu', datetime.now()))
  res = run_experiment(exp)
  res.to_csv(res_path, index=False, header=True)

print('LU Sycl GPU')
for exp in tqdm(exps.experiments_lu_sycl_gpu):
  res_path = os.path.join(results_path, 'exp_{}_{}.csv'.format('lu_sycl_gpu', datetime.now()))
  res = run_experiment(exp)
  res.to_csv(res_path, index=False, header=True)

print('MM OMP')
for exp in tqdm(exps.experiments_mm_omp):
  res_path = os.path.join(results_path, 'exp_{}_{}.csv'.format('mm_omp', datetime.now()))
  res = run_experiment(exp)
  res.to_csv(res_path, index=False, header=True)

print('MM CUDA')
for exp in tqdm(exps.experiments_mm_cuda):
  res_path = os.path.join(results_path, 'exp_{}_{}.csv'.format('mm_cuda', datetime.now()))
  res = run_experiment(exp)
  res.to_csv(res_path, index=False, header=True)

print('MM Sycl CPU')
for exp in tqdm(exps.experiments_mm_sycl_cpu):
  res_path = os.path.join(results_path, 'exp_{}_{}.csv'.format('lu_sycl_cpu', datetime.now()))
  res = run_experiment(exp)
  res.to_csv(res_path, index=False, header=True)

print('MM Sycl GPU')
for exp in tqdm(exps.experiments_mm_sycl_gpu):
  res_path = os.path.join(results_path, 'exp_{}_{}.csv'.format('mm_sycl_gpu', datetime.now()))
  res = run_experiment(exp)
  res.to_csv(res_path, index=False, header=True)