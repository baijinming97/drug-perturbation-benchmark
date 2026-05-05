import torch
import torch.nn.functional as F
from einops import repeat
from diffusers import DDIMScheduler, DDPMScheduler

class Diffusion_Sampler:
    def __init__(self, 
        sampler_type = "DDPM",
        num_train_timesteps=1000, 
        timesteps=50, 
        start=1e-4,
        end=0.02,
        beta_schedule = "linear",
        device=torch.device("cpu"),
        guidance_scale = 5.0,
        uncond = None,
        t_add = None,
        loss_ratio = None
    ):
        self.guidance_scale = guidance_scale
        self.uncond = uncond
        if self.uncond is not None:
            print("cfg activated")
        if t_add is not None:
            print('Enable using exp-to-exp sampler')
            self.t_add = t_add
        else:
            self.t_add = None
        self.device = device
        if sampler_type == "DDPM":
            self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_start=start,beta_end=end,beta_schedule=beta_schedule,timestep_spacing='leading',clip_sample=False,steps_offset=0)
            self.scheduler.set_timesteps(num_train_timesteps)
        elif sampler_type == "DDIM":
            self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_start=start,beta_end=end,beta_schedule=beta_schedule,timestep_spacing='leading',clip_sample=False,set_alpha_to_one=False,steps_offset=0)
            self.scheduler.set_timesteps(timesteps)

        self.loss_ratio = loss_ratio if loss_ratio is not None else 1.0

    def diffusion_loss_fn(self, model, x, x_cond, x_drug, mask=None, cat=False):
        batch_size = x.shape[0]
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, size=(batch_size,1))
        noise = torch.randn_like(x)
        x_t = self.scheduler.add_noise(x, noise, t)
        if mask is not None:
            if cat:
                x_input = torch.stack([x_t,x_cond],dim=-1)
                output = model(x_input, x_drug, t, mask).squeeze(-1)
            else:
                output = model(x_t, x_cond, x_drug, t, mask).squeeze(-1)
        else:
            x_input = torch.stack([x_t,x_cond],dim=-1)
            output = model(x_input, x_drug, t).squeeze(-1)
        loss = F.mse_loss(output, noise, reduction='mean')
        return loss
    
    @torch.no_grad()
    def ddim_sample(self, model, shape, x_cond, x_drug, mask=None, uncond=None, guidance_scale=None, cat=False):
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        x_t = torch.randn(shape).to(self.device)
        batch_size = x_t.shape[0]
        if self.uncond is not None:
            uncond = repeat(self.uncond, 'n -> b n', b=batch_size).to(self.device)
            # mask_uncond=[2 for i in range(batch_size)]

        for t in self.scheduler.timesteps:
            # predict the noise residual
            if mask is not None:
                if cat:
                    x_input = torch.stack([x_t,x_cond],dim=-1)
                    eps_theta = model(x_input, x_drug, t.unsqueeze(0), mask).squeeze(-1)
                else:
                    eps_theta = model(x_t, x_cond, x_drug, t.unsqueeze(0), mask).squeeze(-1)
            else:
                x_input = torch.stack([x_t,x_cond],dim=-1)
                eps_theta = model(x_input, x_drug, t.unsqueeze(0)).squeeze(-1)
            # perform guidance
            if uncond is not None:
                mask_uncond=[1 for _ in range(batch_size)]
                if mask is not None:
                    if cat:
                        x_input = torch.stack([x_t,x_cond],dim=-1)
                        eps_theta_uncond = model(x_input, uncond[:,None,:], t.unsqueeze(0), mask_uncond).squeeze(-1)
                    else:
                        eps_theta_uncond = model(x_t, x_cond, uncond[:,None,:], t.unsqueeze(0), mask_uncond).squeeze(-1)
                else:
                    x_input = torch.stack([x_t,x_cond],dim=-1)
                    eps_theta_uncond = model(x_input, uncond, t.unsqueeze(0)).squeeze(-1)
                eps_theta = eps_theta_uncond + guidance_scale * (eps_theta - eps_theta_uncond)
            x_t = self.scheduler.step(eps_theta, t, x_t).prev_sample
        return x_t
    
    @torch.no_grad()
    def exp2exp(self, model, shape, x_cond, x_drug, mask=None, uncond=None, guidance_scale=None, cat=False):
        if self.t_add is None:
            print('Please set t_add in the config file')
            raise NotImplementedError
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        batch_size = shape[0]
        noise = torch.randn_like(x_cond)
        times = torch.tensor([self.t_add-1]*batch_size).reshape(batch_size,1)
        x_t = self.scheduler.add_noise(x_cond, noise, times)
        
        if self.uncond is not None:
            uncond = repeat(self.uncond, 'n -> b n', b=batch_size).to(self.device)
            # mask_uncond=[2 for i in range(batch_size)]

        for t in self.scheduler.timesteps[-self.t_add:]:
            # predict the noise residual
            if mask is not None:
                if cat:
                    x_input = torch.stack([x_t,x_cond],dim=-1)
                    eps_theta = model(x_input, x_drug, t.unsqueeze(0), mask).squeeze(-1)
                else:
                    eps_theta = model(x_t, x_cond, x_drug, t.unsqueeze(0), mask).squeeze(-1)
            else:
                x_input = torch.stack([x_t,x_cond],dim=-1)
                eps_theta = model(x_input, x_drug, t.unsqueeze(0)).squeeze(-1)
            # perform guidance
            if uncond is not None:
                mask_uncond=[1 for _ in range(batch_size)]
                if mask is not None:
                    if cat:
                        x_input = torch.stack([x_t,x_cond],dim=-1)
                        eps_theta_uncond = model(x_input, uncond[:,None,:], t.unsqueeze(0), mask_uncond).squeeze(-1)
                    else:
                        eps_theta_uncond = model(x_t, x_cond, uncond[:,None,:], t.unsqueeze(0), mask_uncond).squeeze(-1)
                else:
                    x_input = torch.stack([x_t,x_cond],dim=-1)
                    eps_theta_uncond = model(x_input, uncond, t.unsqueeze(0)).squeeze(-1)
                eps_theta = eps_theta_uncond + guidance_scale * (eps_theta - eps_theta_uncond)
            x_t = self.scheduler.step(eps_theta, t, x_t).prev_sample
        return x_t

    def diffusion_loss_fn_unet(self, model, x, x_cond, x_drug, mask):
        batch_size = x.shape[0]
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, size=(batch_size,1))
        noise = torch.randn_like(x)
        x_t = self.scheduler.add_noise(x, noise, t)
        output = model(x_t[:,None,:], x_cond[:,None,:], t, x_drug, mask).squeeze(1)
        loss = F.mse_loss(output, noise, reduction='mean')
        return loss
    
    def recons_loss(self, model, x, x_cond, x_drug, mask=None, cat=False):
        batch_size = x.shape[0]
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, size=(batch_size,1))
        noise = torch.randn_like(x)
        x_t = self.scheduler.add_noise(x, noise, t)
        if mask is not None:
            if cat:
                x_input = torch.stack([x_t,x_cond],dim=-1)
                output = model(x_input, x_drug, t, mask).squeeze(-1)
            else:
                output = model(x_t, x_cond, x_drug, t, mask).squeeze(-1)
        else:
            x_input = torch.stack([x_t,x_cond],dim=-1)
            output = model(x_input, x_drug, t).squeeze(-1)
        original_loss = F.mse_loss(output, x, reduction='mean')
        positive_indicator_original = (x - x_cond > 0).float()
        positive_indicator_generated = (output - x_cond > 0).float()
        binary_cross_entropy_loss_per_sample = torch.nn.BCELoss(reduction='mean')(positive_indicator_generated, positive_indicator_original)
        scaling_factor = original_loss.detach() / binary_cross_entropy_loss_per_sample.detach()
        total_loss = self.loss_ratio*original_loss + (1-self.loss_ratio)*binary_cross_entropy_loss_per_sample*scaling_factor
        return total_loss

    @torch.no_grad()
    def recons_sample(self, model, shape, x_cond, x_drug, mask=None, uncond=None, guidance_scale=None, cat=False):
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        x_t = torch.randn(shape).to(self.device)
        batch_size = x_t.shape[0]
        if self.uncond is not None:
            uncond = repeat(self.uncond, 'n -> b n', b=batch_size).to(self.device)
            # mask_uncond=[2 for i in range(batch_size)]

        for t in self.scheduler.timesteps:
            # predict the noise residual
            if mask is not None:
                if cat:
                    x_input = torch.stack([x_t,x_cond],dim=-1)
                    x_0 = model(x_input, x_drug, t.unsqueeze(0), mask).squeeze(-1)
                else:
                    x_0 = model(x_t, x_cond, x_drug, t.unsqueeze(0), mask).squeeze(-1)
            else:
                x_input = torch.stack([x_t,x_cond],dim=-1)
                x_0 = model(x_input, x_drug, t.unsqueeze(0)).squeeze(-1)
            # perform guidance
            if uncond is not None:
                mask_uncond=[1 for _ in range(batch_size)]
                if mask is not None:
                    if cat:
                        x_input = torch.stack([x_t,x_cond],dim=-1)
                        x_0_uncond = model(x_input, uncond[:,None,:], t.unsqueeze(0), mask_uncond).squeeze(-1)
                    else:
                        x_0_uncond = model(x_t, x_cond, uncond[:,None,:], t.unsqueeze(0), mask_uncond).squeeze(-1)
                else:
                    x_input = torch.stack([x_t,x_cond],dim=-1)
                    x_0_uncond = model(x_input, uncond, t.unsqueeze(0)).squeeze(-1)
                x_0 = x_0_uncond + guidance_scale * (x_0 - x_0_uncond)
            if t>0:
                noise = torch.randn_like(x_0)
                x_t = self.scheduler.add_noise(x_0, noise, t-1)
        return x_0
    
    @torch.no_grad()
    def ddim_sample_unet(self, model, shape, x_cond, x_drug, mask, uncond=None, guidance_scale=None):
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        x_t = torch.randn(shape).to(self.device)
        batch_size = x_t.shape[0]
        if self.uncond is not None:
            uncond = repeat(self.uncond, 'n m -> b n m', b=batch_size).to(self.device)
            mask_uncond=[2 for i in range(batch_size)]

        for t in self.scheduler.timesteps:
            # predict the noise residual
            eps_theta = model(x_t[:,None,:], x_cond[:,None,:], t.unsqueeze(0), x_drug, mask).squeeze(1)
            # perform guidance
            if uncond is not None:
                eps_theta_uncond = model(x_t[:,None,:], x_cond[:,None,:], t.unsqueeze(0), uncond, mask_uncond).squeeze(1)
                eps_theta = eps_theta_uncond + guidance_scale * (eps_theta - eps_theta_uncond)
            x_t = self.scheduler.step(eps_theta, t, x_t).prev_sample
        return x_t