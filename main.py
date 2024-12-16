import torch
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

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


def runs(rad, n_runs, epochs):
    losses = []
    final_losses = []
    for _ in range(n_runs):
        european_call = EuropeanCall(K, r, sigma, T, S, t_sample_size, S_sample_size, rad)
        # Analytical solution
        # c_ = np.array([ [ black_scholes_call(s, K, r, T[1]-t, sigma) for t in t_grid ] for s in s_grid ]).T
        european_call.train(epochs=epochs)
        # european_call.plot(save=False)
        # plt.plot(european_call.test_loss, label='Uniform testing Loss')
        # plt.plot(european_call.pde_test_loss, label='PDE testing Loss')
        # plt.plot(european_call.losses, label='Training Loss')
        # plt.plot(european_call.pde_loss, label='PDE Loss')
        # plt.plot(european_call.boundary_loss, label='Boundary loss')

        # plt.plot(european_call.boundary_loss1, label='Boundary 1')
        # plt.plot(european_call.boundary_loss2, label='Boundary 2')
        # plt.plot(european_call.boundary_loss3, label='Boundary 3')
        # plt.legend()
        losses.append(european_call.test_loss)
        final_losses.append(european_call.test_loss[-1])

    plt.plot(np.mean(losses, axis=0), label=('RAD sampling' if rad else 'Normal sampling'))
    plt.fill_between(np.arange(len(losses[0])), np.min(losses, axis=0), np.max(losses, axis=0), alpha=0.5)


if __name__ == '__main__':
    runs(rad=True, n_runs=1, epochs=5000)
    runs(rad=False, n_runs=1, epochs=5000)

    # plt.title(('Without ' if not use_rad else '') + 'RAD - Loss: ' + str(np.mean(final_losses)))
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    # name = ('no_rad' if not use_rad else 'rad') + '_loss' + str(datetime.now()) + '.png'
    name = 'comparison' + str(datetime.now()) + '.png'
    # plt.savefig('plots/' + name, transparent=True)
    plt.show()
    # european_call.plot_analytical(S, K, r, T, sigma)
