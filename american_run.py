import torch
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

from american_put import AmericanPut

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

# Black-Scholes Parameters
K = 50
r = 0.05
sigma = 0.1
T = [0.0, 1.0]
S = [10, 100]
t_sample_size = 101
S_sample_size = 101

use_rad = False

american_put = AmericanPut(K, r, sigma, T, S, t_sample_size, S_sample_size, use_rad)

american_put.train(epochs=2000)
american_put.plot(save=True)

plt.plot(american_put.losses, label='PDE Loss')
plt.plot(american_put.fb_losses, label='Early Exercise Loss')
plt.title(('Without ' if not use_rad else '') + 'RAD - Loss: ' + str(american_put.losses[-1]))
plt.yscale('log')
plt.legend()
# name = ('no_rad' if not use_rad else 'rad') + '_loss' + str(datetime.now()) + '.png'
# plt.savefig('plots/' + name, transparent=True)
plt.show()

plt.plot(np.linspace(0, 1, 100), american_put.fb(torch.linspace(0, 1, 100).unsqueeze(1)).detach())
plt.show()