import torch
from torch import nn
from PINN import PINN
from tqdm.auto import tqdm
import numpy as np
from scipy.stats import norm


def black_scholes_call(underlying_price, strike_price, interest_rate, days_until_expiration, volatility):
    time_to_expire = days_until_expiration / 365
    d1 = (np.log(underlying_price / strike_price) + time_to_expire * (interest_rate + (volatility ** 2 / 2))) / (
                volatility * np.sqrt(time_to_expire))
    d2 = d1 - (volatility * np.sqrt(time_to_expire))
    call = underlying_price * norm.cdf(d1, 0, 1) - strike_price * np.exp(
        -interest_rate * time_to_expire) * norm.cdf(d2, 0, 1)
    return call


class EuropeanCall:
    def __init__(self, K, r, sigma, T, S, t_sample_size, S_sample_size):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.S = S
        self.t_sample_size = t_sample_size
        self.S_sample_size = S_sample_size

        self.pinn = PINN(2, 16, 1)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.pinn.parameters(), lr=0.01)

        self.t_samples = torch.linspace(T[0], T[1], self.t_sample_size)
        self.S_samples = torch.linspace(S[0], S[1], self.S_sample_size)

        # Boundary: C(0,t) = 0
        self.boundary1 = torch.stack((torch.full((self.t_sample_size,), S[0]), self.t_samples), dim=1).requires_grad_(True)
        # Bourdary: C(S->inf,t) = S-Ke^-r(T-t)
        self.boundary2 = torch.stack((torch.full((self.t_sample_size,), S[1]), self.t_samples), dim=1).requires_grad_(True)
        # Boundary: C(S,T) = max(S-K, 0)
        self.boundary3 = torch.stack((self.S_samples, torch.full((self.S_sample_size,), T[1])), dim=1).requires_grad_(True)

        # Mesh (S,t)
        self.mesh = torch.cartesian_prod(self.S_samples, self.t_samples).requires_grad_(True)

        self.losses = []
        self.data_loss = []

    def loss(self):
        # Boundary losses
        u = self.pinn(self.boundary1)
        loss_boundary1 = torch.squeeze(u).pow(2)

        u = self.pinn(self.boundary2)
        S_inf = self.S[1] - self.K * torch.exp(-self.r * (self.T[1] - self.t_samples))
        loss_boundary2 = (torch.squeeze(u) - S_inf).pow(2)

        u = self.pinn(self.boundary3)
        loss_boundary3 = (torch.squeeze(u) - torch.fmax(self.S_samples - self.K, torch.tensor(0))).pow(2)

        boudary_loss = loss_boundary1 + loss_boundary2 + loss_boundary3

        # PDE loss
        u = self.pinn(self.mesh)
        du = torch.autograd.grad(u, self.mesh, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        dudt, duds = du[:, 0], du[:, 1]
        d2uds2 = torch.autograd.grad(duds, self.mesh, grad_outputs=torch.ones_like(duds), retain_graph=True, create_graph=True)[0][:,1]

        S1 = self.mesh[:, 1]
        pde_loss = (dudt + 0.5 * self.sigma ** 2 * S1 ** 2 * d2uds2 + self.r * S1 * duds - self.r * u).pow(2)

        # data loss
        analytical_solution = black_scholes_call(self.mesh[:, 0].detach(), self.K, self.r, self.T[1] - self.mesh[:, 1].detach(), self.sigma)
        self.data_loss.append((torch.squeeze(u) - analytical_solution).pow(2).mean())

        loss = pde_loss.mean() + boudary_loss.mean()

        return loss

    def train(self, epochs=100):
        for i in tqdm(range(epochs)):
            self.optimizer.zero_grad()

            loss = self.loss()
            self.losses.append(loss.item())

            loss.backward()

            self.optimizer.step()

        return self.pinn

    def plot(self):
        s_grid = np.linspace(self.S[0], self.S[1], self.S_sample_size)
        t_grid = np.linspace(self.T[0], self.T[1], self.t_sample_size)
        s_grid_mesh, t_grid_mesh = np.meshgrid(s_grid, t_grid)

        c = self.pinn(self.mesh).detach().numpy().reshape(t_grid_mesh.shape).T

        plot_surface(s_grid_mesh, t_grid_mesh, c, title='European Call Option')

