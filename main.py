import torch
import torch.nn as nn
from european_call import EuropeanCall
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, qmc
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm.auto import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

# Black-Scholes Parameters
K = 40
r = 0.05
sigma = 0.15
T = [0.0, 1.0]
S = [0, 160]
t_sample_size = 101
S_sample_size = 101

def main():
    european_call = EuropeanCall(K, r, sigma, T, S).train(epochs=1000)
    # Analytical solution
    # c_ = np.array([ [ black_scholes_call(s, K, r, T[1]-t, sigma) for t in t_grid ] for s in s_grid ]).T

    european_call.plot()
