from matplotlib import pyplot as plt
from torch import nn
import torch
import numpy as np
from tqdm import tqdm

from PINN import PINN
from european_call import plot_surface


class AmericanPut:
    def __init__(self, K, r, sigma, T, S, t_sample_size, S_sample_size, use_rad):
        # Black-Scholes Parameters
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.S = S
        self.t_sample_size = t_sample_size
        self.S_sample_size = S_sample_size

        # RAD Parameters
        self.use_rad = use_rad
        self.rad_k = 1
        self.rad_c = 1

        self.boundary_size = 500
        self.mesh_size = int(2500/4)
        self.mesh_big_size = 100000

        self.pinn = PINN(2, 50, 1)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.pinn.parameters(), lr=0.001)

        self.fb = FB(1, 15, 1)
        self.fb_mse_loss = nn.MSELoss()
        self.fb_optimizer = torch.optim.Adam(self.fb.parameters(), lr=0.01)

        self.t_samples = torch.rand(self.boundary_size) * (T[1] - T[0]) + T[0]
        self.S_samples = torch.rand(self.boundary_size) * (S[1] - S[0]) + S[0]

        # Boundary: C(0,t) = 0
        self.boundary1 = self.random_t_tensor(self.boundary_size, T, S[0])
        # Bourdary: C(S->inf,t) = S-Ke^-r(T-t)
        self.boundary2 = self.random_t_tensor(self.boundary_size, T, S[1])
        # Boundary: C(S,T) = max(S-K, 0)
        self.boundary3 = self.random_s_tensor(self.boundary_size, S, T[1])

        # Mesh (S,t)
        self.mesh = self.random_mesh_tensor(self.mesh_size, (self.S[0], self.S[1]), (self.T[0], self.T[1]))

        # Big mesh to sample from
        self.mesh_big = self.random_mesh_tensor(self.mesh_big_size, (self.S[0], self.S[1]), (self.T[0], self.T[1]))

        self.losses = []
        self.fb_losses = []

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
        # u = self.pinn(self.boundary1)
        # loss_boundary1 = self.mse_loss(u, torch.zeros_like(u))

        u = self.pinn(self.boundary2)
        # S_inf = self.S[1] - self.K * torch.exp(-self.r * (self.T[1] - self.boundary2[:, 1]))
        loss_boundary2 = self.mse_loss(u, torch.zeros_like(u))

        u = self.pinn(self.boundary3)
        loss_boundary3 = self.mse_loss(torch.squeeze(u), torch.fmax(self.K - self.boundary3[:, 0], torch.tensor(0)))

        boudary_loss = loss_boundary2 + loss_boundary3

        # RAD
        if self.use_rad and iter > 0 and iter % 50 == 0:
            u = self.pinn(self.mesh_big)
            du = torch.autograd.grad(u, self.mesh_big, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
            dudt, duds = du[:, 0], du[:, 1]
            d2uds2 = torch.autograd.grad(duds, self.mesh_big, grad_outputs=torch.ones_like(duds), retain_graph=True, create_graph=True)[0][:, 1]
            S1 = self.mesh_big[:, 1]

            pde_pdf = (dudt + 0.5 * self.sigma**2 * S1**2 * d2uds2 + self.r * S1 * duds - self.r * torch.squeeze(u)).abs()
            pde_pdf = pde_pdf**self.rad_k / (pde_pdf**self.rad_k).mean() + self.rad_c
            pde_pdf = pde_pdf / pde_pdf.sum()
            sample_idx = np.random.choice(a=len(u), size=self.mesh_size, replace=False, p=pde_pdf.detach().numpy())
            self.mesh = self.mesh_big[sample_idx]

            # show the sampling
            if iter == 1950:
                plt.scatter(self.mesh.detach()[:, 0], self.mesh.detach()[:, 1], s=1)
                plt.savefig('plots/sampling.png', transparent=True)
                plt.show()

            self.boundary1 = self.random_t_tensor(self.boundary_size, self.T, self.S[0])
            self.boundary2 = self.random_t_tensor(self.boundary_size, self.T, self.S[1])
            self.boundary3 = self.random_s_tensor(self.boundary_size, self.S, self.T[1])

        # PDE loss
        u = self.pinn(self.mesh)
        du = torch.autograd.grad(u, self.mesh, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        dudt, duds = du[:, 0], du[:, 1]
        d2uds2 = torch.autograd.grad(duds, self.mesh, grad_outputs=torch.ones_like(duds), retain_graph=True, create_graph=True)[0][:,1]
        S1 = self.mesh[:, 1]

        pde = dudt + 0.5 * self.sigma ** 2 * S1 ** 2 * d2uds2 + self.r * S1 * duds - (self.r * torch.squeeze(u))

        # early = u - torch.max(S1 - self.K, torch.zeros_like(S1))
        # pde_loss = self.mse_loss(pde*early, torch.zeros_like(pde*early))
        pde_loss = self.mse_loss(dudt + 0.5 * self.sigma ** 2 * S1 ** 2 * d2uds2 + self.r * S1 * duds, self.r * torch.squeeze(u))

        # FB loss (early exercise)
        fb_u = self.fb(torch.tensor([self.T[1]], dtype=torch.float))
        fb_init_loss = self.fb_mse_loss(fb_u, torch.tensor([self.K], dtype=torch.float))

        x = torch.stack((torch.squeeze(self.fb(self.t_samples.unsqueeze(1))), self.t_samples), dim=1)
        u = self.pinn(x)
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        duds = du[:, 1]

        fb_boundary3_loss = self.mse_loss(torch.squeeze(u), self.K - torch.squeeze(self.fb(self.t_samples.unsqueeze(1))))
        fb_neu_loss = self.mse_loss(duds, torch.full_like(duds, -1))

        fb_loss = (1/4) * (fb_init_loss + fb_boundary3_loss + fb_neu_loss)

        # data loss
        # analytical_solution = black_scholes_call(self.mesh[:, 0].detach(), self.K, self.r, self.T[1] - self.mesh[:, 1].detach(), self.sigma)
        # self.data_loss.append(self.mse_loss(torch.squeeze(u), analytical_solution).item())

        loss = pde_loss + boudary_loss #+ exercise_loss

        return loss, fb_loss

    def train(self, epochs):
        for i in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            self.fb_optimizer.zero_grad()

            loss, fb_loss = self.loss(i)
            self.losses.append(loss.item())
            self.fb_losses.append(fb_loss.item())

            loss.backward(retain_graph=True)
            fb_loss.backward(retain_graph=True)

            self.optimizer.step()
            self.fb_optimizer.step()

        print('FB-Loss:', fb_loss.item())
        return self.pinn

    def plot_samples(self, points, c='r'):
        plt.scatter(points[:, 0], points[:, 1], c=c, s=1, alpha=0.5)
        plt.show()

    def plot(self, save=False):
        s_grid = np.linspace(self.S[0], self.S[1], self.S_sample_size)
        t_grid = np.linspace(self.T[0], self.T[1], self.t_sample_size)
        s_grid_mesh, t_grid_mesh = np.meshgrid(s_grid, t_grid)

        u_mesh = torch.stack((torch.tensor(s_grid_mesh, dtype=torch.float).flatten(),
                              torch.tensor(t_grid_mesh, dtype=torch.float).flatten()), dim=1).detach()

        c = self.pinn(u_mesh).detach().numpy().reshape(t_grid_mesh.shape)

        plot_surface(s_grid_mesh, t_grid_mesh, c, save=save, angle=45)


class FB(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FB, self).__init__()

        self.tanh = nn.Tanh()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l1.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.l1.weight)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l2.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.l2.weight)

        self.l3 = nn.Linear(hidden_size, output_size)
        self.l3.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.l3.weight)

    def forward(self, x):
        out = self.l1(x)
        out = self.tanh(out)
        out = self.l2(out)
        out = self.tanh(out)
        out = self.l3(out)
        return out
