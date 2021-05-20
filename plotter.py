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
    newdf.append(pd.read_csv(name))
  frame = pd.concat(newdf, axis=0, ignore_index=True)

  return frame


lu_data_df = readExp('lu_data')
lu_func_df = readExp('lu_func')
lu_seq_df = readExp('lu_seq')
lu_sycl_cpu_df = readExp('lu_sycl_cpu')
lu_sycl_gpu_df = readExp('lu_sycl_gpu')
mm_cuda_df = readExp('mm_cuda')
mm_omp_df = readExp('mm_omp')
mm_sycl_cpu_df = readExp('mm_sycl_cpu')
mm_sycl_gpu_df = readExp('mm_sycl_gpu')



mm_omp_df['Performance'] = gflops_mm(mm_omp_df['Matrix Size']) / mm_omp_df['Time']
lu_seq_df['Performance'] = gflops_lu(lu_seq_df['Matrix Size']) / lu_seq_df['Time']

colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow']

def plot(df, x, y, xlabel, ylabel, legendTitle, op, p, destName, perBlock=True):

  group_by = ['Matrix Size', 'Block Size'] if perBlock else ['Matrix Size'] 
  mean = df[(df['Op'] == op) & (df['P'] == p) ].groupby(group_by, as_index=False).mean()

  if perBlock:
    for bs, color in zip([128, 256, 512], colors):
      plot = mean[mean['Block Size'] == bs]
      plt.plot(plot[x], plot[y],  '-x', color=color, label=str(bs))
      plt.legend(title=legendTitle)
  else:
      plt.plot(mean[x], mean[y],  '-x', color=colors[0])

  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.ylim(bottom=0)
  plt.savefig(path.join(plots_dir, f'{destName}.png'))
  plt.cla()


plot(mm_omp_df, 'Matrix Size', 'Time', 'Matrix Size', 'Time (s)', 'Block Size', 1, 1, 'mm_1_time')
plot(mm_omp_df, 'Matrix Size', 'Performance', 'Matrix Size', 'Gflop/s', 'Block Size', 1, 1, 'mm_1_perf')
plot(lu_seq_df, 'Matrix Size', 'Time', 'Matrix Size', 'Time (s)', 'Block Size', 1, 1, 'lu_1_time', False)
plot(lu_seq_df, 'Matrix Size', 'Performance', 'Matrix Size', 'Gflop/s', 'Block Size', 1, 1, 'lu_1_perf', False)
plot(lu_seq_df, 'Matrix Size', 'Time', 'Matrix Size', 'Time (s)', 'Block Size', 2, 1, 'lu_2_time')
plot(lu_seq_df, 'Matrix Size', 'Performance', 'Matrix Size', 'Gflop/s', 'Block Size', 2, 1, 'lu_2_perf')







# fig, subplots = plt.subplots(2, 2)

# df_1 = pd.read_csv(path.join(results_dir, 'exp_lu_seq_2021-05-18 17:16:55.349558.csv'))
# df_1['Performance'] = gflops_lu(df_1['Matrix Size']) / df_1['Time']

# time_mean = df_1.groupby('Matrix Size', as_index=False).mean()

# subplots[0,0].scatter(time_mean['Matrix Size'], time_mean['Time'])
# subplots[0,1].scatter(time_mean['Matrix Size'], time_mean['Performance'])
# subplots[0,1].set_ylim(bottom=0)
# plt.show()