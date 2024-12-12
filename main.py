import torch
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

import european_call
from european_call import EuropeanCall

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

# Black-Scholes Parameters
K = 100
r = 0.05
sigma = 0.1
T = [0.0, 1.0]
S = [80, 160]
t_sample_size = 101
S_sample_size = 101

use_rad = False

if __name__ == '__main__':
    losses = []
    for _ in range(1):
        european_call = EuropeanCall(K, r, sigma, T, S, t_sample_size, S_sample_size, use_rad)
        # Analytical solution
        # c_ = np.array([ [ black_scholes_call(s, K, r, T[1]-t, sigma) for t in t_grid ] for s in s_grid ]).T
        european_call.train(epochs=10000)
        european_call.plot(save=True)
        plt.plot(european_call.losses)
        losses.append(european_call.losses[-1])

    plt.title(('Without ' if not use_rad else '') + 'RAD - Loss: ' + str(np.mean(losses)))
    name = ('no_rad' if not use_rad else 'rad') + '_loss' + str(datetime.now()) + '.png'
    plt.yscale('log')
    plt.savefig('plots/' + name, transparent=True)
    plt.show()
    # european_call.plot_analytical(S, K, r, T, sigma)
