import torch
from torch import nn
from PINN import PINN
from tqdm.auto import tqdm
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class GenericOption:
    def __init__(self, K, r, sigma, T, S, t_sample_size, S_sample_size, use_rad, rad_k=1, rad_c=1):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.S = S
        self.t_sample_size = t_sample_size
        self.S_sample_size = S_sample_size

        self.use_rad = use_rad
        self.rad_k = 2
        self.rad_c = 1

        self.pinn = PINN(2, 16, 1)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.pinn.parameters(), lr=0.001)

        self.t_samples = torch.linspace(T[0], T[1], self.t_sample_size)
        self.S_samples = torch.linspace(S[0], S[1], self.S_sample_size)

        self.boundary_size = 200
        self.mesh_size = 2000
        self.mesh_big_size = 10000

        self.boundary1_weight = 1#1e+2
        self.boundary2_weight = 1e-0
        self.boundary3_weight = 1e-0
        self.pde_weight = 1e+0

        # Boundary: C(0,t) = 0
        self.boundary1_uni = torch.stack((torch.full((self.t_sample_size,), S[0]), self.t_samples), dim=1).requires_grad_(True)
        self.boundary1 = self.random_t_tensor(self.boundary_size, T, S[0])
        # Boundary: C(S->inf,t) = S-Ke^-r(T-t)
        self.boundary2_uni = torch.stack((torch.full((self.t_sample_size,), S[1]), self.t_samples), dim=1).requires_grad_(True)
        self.boundary2 = self.random_t_tensor(self.boundary_size, T, S[1])
        # Boundary: C(S,T) = max(S-K, 0)
        self.boundary3_uni = torch.stack((self.S_samples, torch.full((self.S_sample_size,), T[1])), dim=1).requires_grad_(True)
        self.boundary3 = self.random_s_tensor(self.boundary_size, S, T[1])

        # Mesh (S,t)
        self.mesh = self.random_mesh_tensor(self.mesh_size, (self.S[0], self.S[1]), (self.T[0], self.T[1]))

        # Big mesh to sample from
        self.mesh_big = self.random_mesh_tensor(self.mesh_big_size, (self.S[0], self.S[1]), (self.T[0], self.T[1]))

        # Uniform loss
        self.uniform_mesh = torch.cartesian_prod(torch.linspace(S[0], S[1], 100),
                                                 torch.linspace(T[0], T[1], 100)).requires_grad_(True)

        self.losses = []
        self.test_loss = []
        self.pde_test_loss = []
        self.boundary_loss = []
        self.boundary_loss1 = []
        self.boundary_loss2 = []
        self.boundary_loss3 = []
        self.pde_loss = []
        self.data_loss = []

    def random_mesh_tensor(self, size: int, range_x, range_y):
        return torch.stack((
            torch.rand(size) * (range_x[1] - range_x[0]) + range_x[0],
            torch.rand(size) * (range_y[1] - range_y[0]) + range_y[0]), dim=1
        ).requires_grad_(True)

    def random_t_tensor(self, size: int, trange, S):
        return torch.stack((
            torch.full((size,), S),
            torch.rand(size) * (trange[1] - trange[0]) + trange[0]), dim=1
        ).requires_grad_(True)

    def random_s_tensor(self, size: int, srange, t):
        return torch.stack((
            torch.rand(size) * (srange[1] - srange[0]) + srange[0],
            torch.full((size,), t)), dim=1
        ).requires_grad_(True)

    def pde(self, x):
        u = self.pinn(x)
        du = torch.autograd.grad(u, x,
                                 grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        dudt, duds = du[:, 0], du[:, 1]
        d2uds2 = torch.autograd.grad(duds, x,
                                     grad_outputs=torch.ones_like(duds), retain_graph=True, create_graph=True)[0][:, 1]
        S1 = x[:, 1]

        pde = dudt + 0.5 * self.sigma ** 2 * S1 ** 2 * d2uds2 + self.r * S1 * duds - (self.r * torch.squeeze(u))
        return pde

    def loss(self, iter):
        raise Exception('Not implemented')

    def train(self, epochs):
        raise Exception('Not implemented')

    def plot_samples(self, points, c='r'):
        plt.scatter(points[:, 0], points[:, 1], c=c, s=1, alpha=0.5)
        plt.show()

    def black_scholes_call(self, underlying_price, strike_price, interest_rate, days_until_expiration, volatility):
        time_to_expire = days_until_expiration / 365
        d1 = ((np.log(np.divide(underlying_price, strike_price)) + time_to_expire * (
                    interest_rate + (volatility ** 2 / 2)))
              / (volatility * np.sqrt(time_to_expire)))
        d2 = d1 - (volatility * np.sqrt(time_to_expire))
        call = (underlying_price * norm.cdf(d1, 0, 1)
                - strike_price * np.exp(-interest_rate * time_to_expire) * norm.cdf(d2, 0, 1))
        return call

    def plot_surface(self, x, y, z, title='', save=False, angle=130):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d', elev=30, azim=angle)
        surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
        # fig.colorbar(surf, shrink=0.5, aspect=10)
        ax.set_xlabel('S')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        # ax.set_title(title)
        if save:
            plt.savefig('plots/european_call.png', transparent=True)
        plt.show()

    def plot_analytical(self):
        s_grid = np.linspace(self.S[0], self.S[1], 50)
        t_grid = np.linspace(self.T[0], self.T[1], 50)
        s_grid_mesh, t_grid_mesh = np.meshgrid(s_grid, t_grid)
        bs = self.black_scholes_call(s_grid_mesh, self.K, self.r, self.T[1], self.sigma)
        self.plot_surface(s_grid_mesh, t_grid_mesh, bs, '')

    def plot(self, save=False):
        s_grid = np.linspace(self.S[0], self.S[1], self.S_sample_size)
        t_grid = np.linspace(self.T[0], self.T[1], self.t_sample_size)
        s_grid_mesh, t_grid_mesh = np.meshgrid(s_grid, t_grid)

        u_mesh = torch.stack((torch.tensor(s_grid_mesh, dtype=torch.float).flatten(),
                              torch.tensor(t_grid_mesh, dtype=torch.float).flatten()), dim=1).detach()

        c = self.pinn(u_mesh).detach().numpy().reshape(t_grid_mesh.shape)

        self.plot_surface(s_grid_mesh, t_grid_mesh, c, save=save)

