import os
from os import path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob

from pandas.core.frame import DataFrame

dir = path.dirname(__file__)
plots_dir = path.join(dir, 'plots')
results_dir = path.join(dir, 'results')
os.makedirs(path.join(dir, 'plots'), exist_ok=True)

def gflops_mm(mat_size): return 2 * (mat_size ** 3) * 1e-9
def gflops_lu(mat_size): return 2/3 * (mat_size ** 3) * 1e-9

def readExp(folder):
  newdf = [] 
  for name in glob.glob(path.join(results_dir, f'{folder}/*.csv')):
    newdf = pd.read_csv(name)
    newdf.append(newdf)

  return pd.concat(newdf, axis=0, ignore_index=True)


lu_data_df = readExp('lu_data')
lu_func_df = readExp('lu_func')
lu_seq_df = readExp('lu_seq')
lu_sycl_cpu_df = readExp('lu_sycl_cpu')
lu_sycl_gpu_df = readExp('lu_sycl_gpu')
mm_cuda_df = readExp('mm_cuda')
mm_omp_df = readExp('mm_omp')
mm_sycl_cpu_df = readExp('mm_sycl_cpu')
mm_sycl_gpu_df = readExp('mm_sycl_gpu')













# fig, subplots = plt.subplots(2, 2)

# df_1 = pd.read_csv(path.join(results_dir, 'exp_lu_seq_2021-05-18 17:16:55.349558.csv'))
# df_1['Performance'] = gflops_lu(df_1['Matrix Size']) / df_1['Time']

# time_mean = df_1.groupby('Matrix Size', as_index=False).mean()

# subplots[0,0].scatter(time_mean['Matrix Size'], time_mean['Time'])
# subplots[0,1].scatter(time_mean['Matrix Size'], time_mean['Performance'])
# subplots[0,1].set_ylim(bottom=0)
# plt.show()