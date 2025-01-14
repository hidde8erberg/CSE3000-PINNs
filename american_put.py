from matplotlib import pyplot as plt
from torch import nn
import torch
import numpy as np
from tqdm import tqdm

from PINN import FB
from generic_option import GenericOption


class AmericanPut(GenericOption):

    def __init__(self, K, r, sigma, T, S, t_sample_size, S_sample_size, use_rad, rad_k=1, rad_c=1, rad_interval=50):
        super().__init__(K, r, sigma, T, S, t_sample_size, S_sample_size, use_rad, rad_k, rad_c, lr=0.001)

        self.rad_interval = rad_interval

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
        if self.use_rad and iter > 0 and iter % self.rad_interval == 0:
            u = self.pde(self.mesh_big)
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
        u = self.pde(self.mesh)
        pde_loss = self.mse_loss(u, torch.zeros_like(u))

        # FB loss (early exercise)
        fb_u = self.fb(torch.tensor([self.T[1]], dtype=torch.float))
        fb_init_loss = self.fb_mse_loss(fb_u, torch.tensor([self.K], dtype=torch.float))

        x = torch.stack((torch.squeeze(self.fb(self.t_samples.unsqueeze(1))), self.t_samples), dim=1)
        u = self.pinn(x)
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        duds = du[:, 1]

        fb_boundary3_loss = self.mse_loss(torch.squeeze(u), self.K - torch.squeeze(self.fb(self.t_samples.unsqueeze(1))))
        fb_neu_loss = self.mse_loss(duds, torch.full_like(duds, -1))

        fb_loss = fb_init_loss + fb_boundary3_loss + fb_neu_loss

        u = self.pde(self.uniform_mesh)
        pde_uni_loss = self.mse_loss(torch.squeeze(u), torch.zeros_like(u))
        self.pde_loss.append(pde_uni_loss.item())
        u = self.pinn(self.boundary1_uni)
        b1_test_loss = self.mse_loss(u, torch.full_like(u, self.K))
        self.boundary_loss1.append(b1_test_loss.item())
        u = self.pinn(self.boundary2_uni)
        b2_test_loss = self.mse_loss(u, torch.zeros_like(u))
        self.boundary_loss2.append(b2_test_loss.item())
        u = self.pinn(self.boundary3_uni)
        b3_test_loss = self.mse_loss(torch.squeeze(u), torch.fmax(self.K - self.boundary3_uni[:, 0], torch.tensor(0)))
        self.boundary_loss3.append(b3_test_loss.item())

        self.test_loss.append(pde_uni_loss.item()+b1_test_loss.item()+b2_test_loss.item()+b3_test_loss.item())

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
