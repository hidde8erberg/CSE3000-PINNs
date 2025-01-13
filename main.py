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
t_sample_size = 500
S_sample_size = 500


def runs(rad, n_runs, epochs, rad_k=2, rad_c=1, rad_interval=50):
    losses = []
    final_losses = []
    for i in range(n_runs):
        european_call = EuropeanCall(K, r, sigma, T, S, t_sample_size, S_sample_size, rad,
                                     rad_k=rad_k, rad_c=rad_c, rad_interval=rad_interval)
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

    avg_losses = np.mean(losses, axis=0)
    std_losses = np.std(losses, axis=0)
    plt.plot(avg_losses, label=(f"RAD sampling - interval={rad_interval}" if rad else 'Normal sampling'))
    plt.fill_between(np.arange(len(losses[0])), avg_losses - std_losses, avg_losses + std_losses, alpha=0.5)


if __name__ == '__main__':
    # runs(rad=True, n_runs=2, epochs=16000, rad_k=2, rad_c=0, rad_interval=10)
    # runs(rad=True, n_runs=3, epochs=15000, rad_k=2, rad_c=0, rad_interval=50)
    # runs(rad=True, n_runs=3, epochs=15000, rad_k=2, rad_c=0, rad_interval=100)
    # runs(rad=True, n_runs=3, epochs=15000, rad_k=2, rad_c=0, rad_interval=500)
    #
    # plt.title('Comparison of RAD intervals - with k=2 c=0')
    # plt.xlabel('Iterations')
    # plt.ylabel('MSE Loss')
    # plt.yscale('log')
    # plt.legend()
    # name = 'comparison' + str(datetime.now()) + '.png'
    # plt.savefig('plots/' + name, transparent=False)
    # plt.show()
    european_call = EuropeanCall(K, r, sigma, T, S, t_sample_size, S_sample_size, True,
                                 rad_k=1, rad_c=1, rad_interval=50)
    european_call.plot_analytical()
