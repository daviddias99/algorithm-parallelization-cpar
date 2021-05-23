import os
import subprocess
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