import torch
from matplotlib import pyplot as plt

from european_call import EuropeanCall

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

# Black-Scholes Parameters
K = 40
r = 0.05
sigma = 0.15
T = [0.0, 1.0]
S = [0, 160]
t_sample_size = 101
S_sample_size = 101

if __name__ == '__main__':
    european_call = EuropeanCall(K, r, sigma, T, S, t_sample_size, S_sample_size)
    # Analytical solution
    # c_ = np.array([ [ black_scholes_call(s, K, r, T[1]-t, sigma) for t in t_grid ] for s in s_grid ]).T

    european_call.train(epochs=1000)

    european_call.plot()

    plt.plot(european_call.data_loss)
    plt.show()
