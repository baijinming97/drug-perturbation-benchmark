import tqdm
import numpy as np
import torch

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, model, x, x_cond, x_drug, mask=None):
        rnd_normal = torch.randn([x.shape[0], 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x) * sigma
        if mask is None:
            D_yn = model(x + n, x_cond, x_drug, sigma)
        else:
            D_yn = model(x + n, x_cond, x_drug, sigma, mask)
        loss = weight * ((D_yn - x) ** 2)
        return loss.mean()

class EDM_classification_Loss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, loss_ratio = 0.8):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.loss_ratio = loss_ratio

    def __call__(self, model, x, x_cond, x_drug, mask=None):
        rnd_normal = torch.randn([x.shape[0], 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x) * sigma
        if mask is None:
            D_yn = model(x + n, x_cond, x_drug, sigma)
        else:
            D_yn = model(x + n, x_cond, x_drug, sigma, mask)
        # initial loss
        original_loss = (weight * ((D_yn - x) ** 2)).mean()
        # binary loss
        positive_indicator_original = (x - x_cond > 0).float()
        positive_indicator_generated = (D_yn - x_cond > 0).float()
        binary_cross_entropy_loss_per_sample = (weight * (torch.nn.BCELoss(reduction='none')(positive_indicator_generated, positive_indicator_original))).mean()
        scaling_factor = original_loss.detach() / binary_cross_entropy_loss_per_sample.detach()
        # combine two losses
        total_loss = self.loss_ratio*original_loss + (1-self.loss_ratio)*binary_cross_entropy_loss_per_sample*scaling_factor
        return total_loss

class EDM_Sampler:
    def __init__(self, 
        num_steps=18,
        sigma_min=0.002, 
        sigma_max=80,
        device=torch.device("cpu"), 
        rho=7,
        S_churn=0, 
        S_min=0, 
        S_max=float('inf'), 
        S_noise=1,
        cls_loss=False,
    ):
        self.num_steps = num_steps
        self.device = device
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min 
        self.S_max = S_max
        self.S_noise = S_noise
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        self.t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.t_steps1 = torch.cat([self.t_steps, torch.zeros_like(self.t_steps[:1])]) # t_N = 0
        if cls_loss:
            self.edm_loss = EDM_classification_Loss()
        else:
            self.edm_loss = EDMLoss()

    @torch.no_grad()
    def edm_sampler(self, model, shape, x_cond, x_drug, mask=None):
        # Main sampling loop.
        x_next = torch.randn(shape, dtype=torch.float64, device=self.device) * self.t_steps1[0]
        for i, (t_cur, t_next) in enumerate(zip(self.t_steps1[:-1], self.t_steps1[1:])): # 0, ..., N-1
            x_cur = x_next
            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            if mask is None:
                denoised = model(x_hat, x_cond, x_drug, t_hat).to(torch.float64)
            else:
                denoised = model(x_hat, x_cond, x_drug, t_hat, mask).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                if mask is None:
                    denoised = model(x_next, x_cond, x_drug, t_next).to(torch.float64)
                else:
                    denoised = model(x_next, x_cond, x_drug, t_next, mask).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

