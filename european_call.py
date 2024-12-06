import torch
from torch import nn
from PINN import PINN
from tqdm.auto import tqdm
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def black_scholes_call(underlying_price, strike_price, interest_rate, days_until_expiration, volatility):
    time_to_expire = days_until_expiration / 365
    d1 = (np.log(np.divide(underlying_price, strike_price)) + time_to_expire * (interest_rate + (volatility ** 2 / 2))) / (
                volatility * np.sqrt(time_to_expire))
    d2 = d1 - (volatility * np.sqrt(time_to_expire))
    call = underlying_price * norm.cdf(d1, 0, 1) - strike_price * np.exp(
        -interest_rate * time_to_expire) * norm.cdf(d2, 0, 1)
    return call


def plot_surface(x, y, z, title='', save=False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d', elev=30, azim=130)
    surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    # fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_xlabel('S')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    # ax.set_title(title)
    if save:
        plt.savefig('plots/european_call.png', transparent=True)
    plt.show()


class EuropeanCall:
    def __init__(self, K, r, sigma, T, S, t_sample_size, S_sample_size, use_rad):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.S = S
        self.t_sample_size = t_sample_size
        self.S_sample_size = S_sample_size

        self.use_rad = use_rad = True
        self.rad_k = 1
        self.rad_c = 1

        self.pinn = PINN(2, 50, 1)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.pinn.parameters(), lr=0.001)

        self.t_samples = torch.linspace(T[0], T[1], self.t_sample_size)
        self.S_samples = torch.linspace(S[0], S[1], self.S_sample_size)

        self.boundary_size = 500
        self.mesh_size = 2500
        self.mesh_big_size = 100000

        # Boundary: C(0,t) = 0
        self.boundary1_uni = torch.stack((torch.full((self.t_sample_size,), S[0]), self.t_samples), dim=1).requires_grad_(True)
        self.boundary1 = self.random_t_tensor(self.boundary_size, T, S[0])
        # Bourdary: C(S->inf,t) = S-Ke^-r(T-t)
        self.boundary2_uni = torch.stack((torch.full((self.t_sample_size,), S[1]), self.t_samples), dim=1).requires_grad_(True)
        self.boundary2 = self.random_t_tensor(self.boundary_size, T, S[1])
        # Boundary: C(S,T) = max(S-K, 0)
        self.boundary3_uni = torch.stack((self.S_samples, torch.full((self.S_sample_size,), T[1])), dim=1).requires_grad_(True)
        self.boundary3 = self.random_s_tensor(self.boundary_size, S, T[1])

        # Mesh (S,t)
        self.mesh_uni = torch.cartesian_prod(self.S_samples, self.t_samples).requires_grad_(True)
        self.mesh = self.random_mesh_tensor(self.mesh_size, (self.S[0], self.S[1]), (self.T[0], self.T[1]))

        # Big mesh to sample from
        self.mesh_big = self.random_mesh_tensor(self.mesh_big_size, (self.S[0], self.S[1]), (self.T[0], self.T[1]))

        self.losses = []
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

    def loss(self, iter):
        # Boundary losses
        u = self.pinn(self.boundary1)
        loss_boundary1 = self.mse_loss(u, torch.zeros_like(u))

        u = self.pinn(self.boundary2)
        S_inf = self.S[1] - self.K * torch.exp(-self.r * (self.T[1] - self.boundary2[:, 1]))
        loss_boundary2 = self.mse_loss(torch.squeeze(u), S_inf)

        u = self.pinn(self.boundary3)
        loss_boundary3 = self.mse_loss(torch.squeeze(u), torch.fmax(self.boundary3[:, 0] - self.K, torch.tensor(0)))

        boudary_loss = loss_boundary1 + loss_boundary2 + loss_boundary3

        # RAD
        if self.use_rad and iter > 0 and iter % 25 == 0:
            u = self.pinn(self.mesh_big)
            du = torch.autograd.grad(u, self.mesh_big, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            dudt, duds = du[:, 0], du[:, 1]
            d2uds2 = torch.autograd.grad(duds, self.mesh_big, grad_outputs=torch.ones_like(duds), retain_graph=True, create_graph=True)[0][:, 1]
            S1 = self.mesh_big[:, 1]

            pde_pdf = (dudt + 0.5 * self.sigma ** 2 * S1 ** 2 * d2uds2 + self.r * S1 * duds - self.r * torch.squeeze(u)).abs()
            pde_pdf = pde_pdf**self.rad_k / (pde_pdf**self.rad_k).mean() + self.rad_c
            pde_pdf = pde_pdf / pde_pdf.sum()
            sample_idx = np.random.choice(a=len(u), size=self.mesh_size, replace=False, p=pde_pdf.detach().numpy())
            self.mesh = self.mesh_big[sample_idx]

            self.boundary1 = self.random_t_tensor(self.boundary_size, self.T, self.S[0])
            self.boundary2 = self.random_t_tensor(self.boundary_size, self.T, self.S[1])
            self.boundary3 = self.random_s_tensor(self.boundary_size, self.S, self.T[1])

        # PDE loss
        u = self.pinn(self.mesh)
        du = torch.autograd.grad(u, self.mesh, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        dudt, duds = du[:, 0], du[:, 1]
        d2uds2 = torch.autograd.grad(duds, self.mesh, grad_outputs=torch.ones_like(duds), retain_graph=True, create_graph=True)[0][:,1]
        S1 = self.mesh[:, 1]

        pde_loss = self.mse_loss(dudt + 0.5 * self.sigma ** 2 * S1 ** 2 * d2uds2 + self.r * S1 * duds, self.r * torch.squeeze(u))

        # data loss
        # analytical_solution = black_scholes_call(self.mesh[:, 0].detach(), self.K, self.r, self.T[1] - self.mesh[:, 1].detach(), self.sigma)
        # self.data_loss.append(self.mse_loss(torch.squeeze(u), analytical_solution).item())

        loss = pde_loss + boudary_loss

        return loss

    def train(self, epochs):
        for i in tqdm(range(epochs)):
            self.optimizer.zero_grad()

            loss = self.loss(i)
            self.losses.append(loss.item())

            loss.backward(retain_graph=True)

            self.optimizer.step()

        return self.pinn

    def plot_samples(self, points, c='r'):
        plt.scatter(points[:, 0], points[:, 1], c=c, s=1, alpha=0.5)
        plt.show()

    def plot(self):
        s_grid = np.linspace(self.S[0], self.S[1], self.S_sample_size)
        t_grid = np.linspace(self.T[0], self.T[1], self.t_sample_size)
        s_grid_mesh, t_grid_mesh = np.meshgrid(s_grid, t_grid)

        u_mesh = torch.stack((torch.tensor(s_grid_mesh, dtype=torch.float).flatten(),
                              torch.tensor(t_grid_mesh, dtype=torch.float).flatten()), dim=1).detach()

        c = self.pinn(u_mesh).detach().numpy().reshape(t_grid_mesh.shape)

        plot_surface(s_grid_mesh, t_grid_mesh, c, save=False)
