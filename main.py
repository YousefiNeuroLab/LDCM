import os
import numpy as np
import pandas as pd
from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.utils import resample
import seaborn as sns
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.metrics import roc_curve, roc_auc_score

# Configuration parameters
dim_state = 1
dim_obs = 1
simulatiom_mode = 1
update_params_mode = [1, 1, 2, 1] #If 1 for any of spaces it will not be updated,Shape of: [A@B,C@D,Q,R]
R0 = np.array([100])
R = np.array([0.005])
n_steps = 360
num_particles = 200
num_trials = 23
max_iter = 3
mode = 1

# Initialize and run the EM algorithm
em = EMAlgorithm(dim_state, dim_obs, simulatiom_mode, update_params_mode, mode, R, R0, n_steps, num_particles, num_trials, max_iter)
ans_x = []
ans_y = []
for i in range(7):
    em.run_nn()
    em.run_em()
    x, y = em.result()
    ans_x.append(x)
    ans_y.append(y)
print(f"Average test accuracy with particle: {np.mean(ans_x):.4f}")
print(f"Average test accuracy: {np.mean(ans_y):.4f}")
print("x",ans_x)
print("y",ans_y)
