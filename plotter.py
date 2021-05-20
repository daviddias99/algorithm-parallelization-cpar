import os
from os import path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dir = path.dirname(__file__)
plots_dir = path.join(dir, 'plots')
results_dir = path.join(dir, 'results')
os.makedirs(path.join(dir, 'plots'), exist_ok=True)

def gflops_mm(mat_size): return 2 * (mat_size ** 3) * 1e-9
def gflops_lu(mat_size): return 2/3 * (mat_size ** 3) * 1e-9

fig, subplots = plt.subplots(2, 2)

df_1 = pd.read_csv(path.join(results_dir, 'exp_lu_seq_2021-05-18 17:16:55.349558.csv'))
df_1['Performance'] = gflops_lu(df_1['Matrix Size']) / df_1['Time']

time_mean = df_1.groupby('Matrix Size', as_index=False).mean()

subplots[0,0].scatter(time_mean['Matrix Size'], time_mean['Time'])
subplots[0,1].scatter(time_mean['Matrix Size'], time_mean['Performance'])
subplots[0,1].set_ylim(bottom=0)
plt.show()