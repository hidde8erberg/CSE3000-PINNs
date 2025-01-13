import torch
from tqdm.auto import tqdm
import numpy as np
from generic_option import GenericOption


class EuropeanCall(GenericOption):

    def __init__(self, K, r, sigma, T, S, t_sample_size, S_sample_size, use_rad, rad_k=1, rad_c=1, rad_interval=50):
        super().__init__(K, r, sigma, T, S, t_sample_size, S_sample_size, use_rad, rad_k, rad_c)

        self.rad_interval = rad_interval

    def loss(self, iter):
        # RAD
        if self.use_rad and iter > 0 and iter % self.rad_interval == 0:
            pde_pdf = self.pde(self.mesh_big).abs()
            pde_pdf = pde_pdf**self.rad_k / (pde_pdf**self.rad_k).mean() + self.rad_c
            pde_pdf = pde_pdf / pde_pdf.sum()
            sample_idx = np.random.choice(a=len(pde_pdf),
                                          size=self.mesh_size,
                                          replace=False,
                                          p=pde_pdf.detach().numpy())
            self.mesh = self.mesh_big[sample_idx]

            # show the sampling
            # if iter == n_iter-rad_interval:
                # plt.scatter(self.mesh.detach()[:, 0], self.mesh.detach()[:, 1], s=1)
                # plt.savefig('plots/sampling.png', transparent=True)
                # plt.show()

            # self.boundary1 = self.random_t_tensor(self.boundary_size, self.T, self.S[0])
            # self.boundary2 = self.random_t_tensor(self.boundary_size, self.T, self.S[1])
            # self.boundary3 = self.random_s_tensor(self.boundary_size, self.S, self.T[1])

        # PDE loss
        pde = self.pde(self.mesh)
        pde_loss = self.pde_weight * self.mse_loss(pde, torch.zeros_like(pde))

        # Boundary losses
        u = self.pinn(self.boundary1)
        loss_boundary1 = self.boundary1_weight * self.mse_loss(u, torch.zeros_like(u))

        u = self.pinn(self.boundary2)
        S_inf = self.S[1] - self.K * torch.exp(-self.r * (self.T[1] - self.boundary2[:, 1]))
        loss_boundary2 = self.boundary2_weight * self.mse_loss(torch.squeeze(u), S_inf)

        u = self.pinn(self.boundary3)
        loss_boundary3 = self.boundary3_weight * self.mse_loss(torch.squeeze(u),
                                                               torch.fmax(self.boundary3[:, 0] - self.K,
                                                                          torch.tensor(0)))

        boundary_loss = (loss_boundary1 + loss_boundary2 + loss_boundary3)

        # data loss
        # analytical_solution = black_scholes_call(self.mesh[:, 0].detach(), self.K, self.r, self.T[1] - self.mesh[:, 1].detach(), self.sigma)
        # self.data_loss.append(self.mse_loss(torch.squeeze(u), analytical_solution).item())

        # Test loss
        test_loss = torch.norm(self.pde(self.uniform_mesh), p=2)
        self.pde_test_loss.append(test_loss.item())
        test_loss += (torch.norm(self.pde(self.boundary1_uni), p=2)
                      + torch.norm(self.pde(self.boundary2_uni), p=2)
                      + torch.norm(self.pde(self.boundary3_uni), p=2))
        self.test_loss.append(test_loss.item() + boundary_loss.item())

        self.boundary_loss.append(boundary_loss.item())
        self.boundary_loss1.append(loss_boundary1.item())
        self.boundary_loss2.append(loss_boundary2.item())
        self.boundary_loss3.append(loss_boundary3.item())

        self.pde_loss.append(pde_loss.item())

        loss = pde_loss + boundary_loss

        return loss

    def train(self, epochs):
        for i in tqdm(range(epochs)):
            self.optimizer.zerok_grad()

            loss = self.loss(i)
            self.losses.append(loss.item())

            loss.backward(retain_graph=True)

            self.optimizer.step()

        return self.pinn
