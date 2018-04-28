from scipy.io import loadmat
import os
import numpy as np


home = os.path.expanduser('~')
data_home = home + '/Documents/research/matsc_project-master/results/new_results/'
M1 = loadmat(data_home + 'matsc_result_mat_files.mat')
M2 = loadmat(data_home + 'trans_long_result_mat_files.mat')

M3 = loadmat(data_home + 'best_tables.mat')
M4 = loadmat(data_home + 'best_tables_trans_long.mat')

print()
