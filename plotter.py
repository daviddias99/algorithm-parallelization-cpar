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

colors = ['blue', 'red', 'green', 'orange', 'purple']

# MM Seq
def plotMMSeq(df, x, y, xlabel, ylabel, legendTitle, op, p, destName, perBlock=True):

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


plotMMSeq(mm_omp_df, 'Matrix Size', 'Time', 'Matrix Size', 'Time (s)', 'Block Size', 1, 1, 'mm_1_time')
plotMMSeq(mm_omp_df, 'Matrix Size', 'Performance', 'Matrix Size', 'Gflop/s', 'Block Size', 1, 1, 'mm_1_perf')

# LU Seq
group_by = ['Matrix Size', 'Block Size'] 
mean = lu_seq_df[(lu_seq_df['Op'] == 2) & (lu_seq_df['P'] == 1) ].groupby(group_by, as_index=False).mean()

for bs, color in zip([128, 256, 512], colors):
  plot = mean[mean['Block Size'] == bs]
  plt.plot(plot['Matrix Size'], plot['Time'],  '-x', color=color, label=f'Blocked ({bs})')

group_by = ['Matrix Size'] 
mean = lu_seq_df[(lu_seq_df['Op'] == 1) & (lu_seq_df['P'] == 1) ].groupby(group_by, as_index=False).mean()
plt.plot(mean['Matrix Size'], mean['Time'],  '-x', color=colors[-1], label=f'Naive ({bs})')
plt.legend(title='Operation')
plt.ylabel('Time (s)')
plt.xlabel('Matrix Size')
plt.ylim(bottom=0)
plt.savefig(path.join(plots_dir, f'lu_1_2_time.png'))
plt.cla()

group_by = ['Matrix Size', 'Block Size'] 
mean = lu_seq_df[(lu_seq_df['Op'] == 2) & (lu_seq_df['P'] == 1) ].groupby(group_by, as_index=False).mean()

for bs, color in zip([128, 256, 512], colors):
  plot = mean[mean['Block Size'] == bs]
  plt.plot(plot['Matrix Size'], plot['Performance'],  '-x', color=color, label=f'Blocked ({bs})')

group_by = ['Matrix Size'] 
mean = lu_seq_df[(lu_seq_df['Op'] == 1) & (lu_seq_df['P'] == 1) ].groupby(group_by, as_index=False).mean()
plt.plot(mean['Matrix Size'], mean['Performance'],  '-x', color=colors[-1], label=f'Naive')
plt.legend(title='Operation')
plt.ylabel('Performance (Gflop/s)')
plt.xlabel('Matrix Size')
plt.ylim(bottom=0)
plt.savefig(path.join(plots_dir, f'lu_1_2_perf.png'))
plt.cla()








