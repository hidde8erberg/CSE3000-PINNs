from matplotlib import pyplot as plt
from torch import nn
import torch
import numpy as np
from tqdm import tqdm

from PINN import FB
from generic_option import GenericOption


class AmericanPut(GenericOption):

    def __init__(self, K, r, sigma, T, S, t_sample_size, S_sample_size, use_rad, rad_k=1, rad_c=1):
        super().__init__(K, r, sigma, T, S, t_sample_size, S_sample_size, use_rad, rad_k, rad_c)

        self.fb = FB(1, 8, 1)
        self.fb_mse_loss = nn.MSELoss()
        self.fb_optimizer = torch.optim.Adam(self.fb.parameters(), lr=0.01)

        self.fb_losses = []

    def loss(self, iter):
        # Boundary losses

        # Boundary S = 0
        u = self.pinn(self.boundary1)
        loss_boundary1 = self.mse_loss(u, torch.full_like(u, self.K))

        # Boundary S -> inf
        u = self.pinn(self.boundary2)
        loss_boundary2 = self.mse_loss(u, torch.zeros_like(u))

        # Initial condition at T
        u = self.pinn(self.boundary3)
        loss_boundary3 = self.mse_loss(torch.squeeze(u), torch.fmax(self.K - self.boundary3[:, 0], torch.tensor(0)))

        boudary_loss = loss_boundary1 + loss_boundary2 + loss_boundary3

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

        # PDE loss
        u = self.pinn(self.mesh)
        du = torch.autograd.grad(u, self.mesh, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        dudt, duds = du[:, 0], du[:, 1]
        d2uds2 = torch.autograd.grad(duds, self.mesh, grad_outputs=torch.ones_like(duds), retain_graph=True, create_graph=True)[0][:,1]
        S1 = self.mesh[:, 1]

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


        # data loss
        # analytical_solution = black_scholes_call(self.mesh[:, 0].detach(), self.K, self.r, self.T[1] - self.mesh[:, 1].detach(), self.sigma)
        # self.data_loss.append(self.mse_loss(torch.squeeze(u), analytical_solution).item())

        test_loss = torch.norm(self.pde(self.uniform_mesh), p=2)
        self.test_loss.append(test_loss.item())

        loss = pde_loss + boudary_loss

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

        return self.pinn
