import torch
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

from american_put import AmericanPut

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

# Black-Scholes Parameters
K = 100
r = 0.05
sigma = 0.1
T = [0.0, 1.0]
S = [80, 160]
t_sample_size = 500
S_sample_size = 500

use_rad = False

american_put = AmericanPut(K, r, sigma, T, S, t_sample_size, S_sample_size, True,
                                         rad_k=1, rad_c=1, rad_interval=50)

american_put.train(epochs=10000)
plt.plot(american_put.pde_test_loss, label='PDE Loss')
plt.plot(american_put.test_loss, label='test Loss')
plt.plot(american_put.fb_losses, label='Early Exercise Loss')
plt.plot(american_put.boundary_loss1, label='Test Loss boundary 1')
plt.plot(american_put.boundary_loss2, label='Test Loss boundary 2')
plt.plot(american_put.boundary_loss3, label='Test Loss boundary 3')
plt.yscale('log')
plt.legend()
# name = ('no_rad' if not use_rad else 'rad') + '_loss' + str(datetime.now()) + '.png'
# plt.savefig('plots/' + name, transparent=True)
plt.show()

# plt.plot(np.linspace(0, 1, 100), american_put.fb(torch.linspace(0, 1, 100).unsqueeze(1)).detach())
# plt.show()