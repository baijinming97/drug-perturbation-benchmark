import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import NegativeBinomial, normal

def EDM_loss_func(sampler, model, batch):
    loss = sampler.edm_loss(model, batch[0], batch[1], batch[2])
    return loss

def EDM_generator(sampler, model, batch):
    y_pred = sampler.edm_sampler(model, batch[0].shape, batch[1], batch[2]).cpu()
    return y_pred

def EDM_Cross_loss_func(sampler, model, batch):
    loss = sampler.edm_loss(model, batch[0], batch[1], batch[2], batch[3])
    return loss

def EDM_Cross_generator(sampler, model, batch):
    y_pred = sampler.edm_sampler(model, batch[0].shape, batch[1], batch[2], batch[3]).cpu()
    return y_pred

def Ada_loss_func(sampler, model, batch):
    loss = sampler.diffusion_loss_fn(model, batch[0], batch[1], batch[2])
    return loss

def Ada_generator(sampler, model, batch):
    y_pred = sampler.ddim_sample(model, batch[0].shape, batch[1], batch[2]).cpu()
    return y_pred

def DirectAda_loss_func(sampler, model, batch):
    output = model(batch[1], batch[2]).squeeze(-1)
    loss = F.mse_loss(output, batch[0], reduction='mean')
    return loss

def DirectAda_generator(sampler, model, batch):
    with torch.no_grad():
        y_pred = model(batch[1], batch[2]).squeeze(-1).cpu()
    return y_pred

def CatCross_loss_func(sampler, model, batch):
    loss = sampler.diffusion_loss_fn(model, batch[0], batch[1], batch[2], batch[3], cat=True)
    return loss

def CatCross_generator(sampler, model, batch):
    y_pred = sampler.ddim_sample(model, batch[0].shape, batch[1], batch[2], batch[3], cat=True).cpu()
    return y_pred

def Cross_loss_func(sampler, model, batch):
    loss = sampler.diffusion_loss_fn(model, batch[0], batch[1], batch[2], batch[3])
    return loss

def Cross_generator(sampler, model, batch):
    y_pred = sampler.ddim_sample(model, batch[0].shape, batch[1], batch[2], batch[3]).cpu()
    return y_pred

def Cross_loss_func_recons(sampler, model, batch):
    loss = sampler.recons_loss(model, batch[0], batch[1], batch[2], batch[3])
    return loss

def Cross_generator_recons(sampler, model, batch):
    y_pred = sampler.recons_sample(model, batch[0].shape, batch[1], batch[2], batch[3]).cpu()
    return y_pred

def Crossexp2exp_generator(sampler, model, batch):
    y_pred = sampler.exp2exp(model, batch[0].shape, batch[1], batch[2], batch[3]).cpu()
    return y_pred

def DirectCross_loss_func(sampler, model, batch):
    output = model(batch[1], batch[2], batch[3]).squeeze(-1)
    loss = F.mse_loss(output, batch[0], reduction='mean')
    return loss

def DirectCross_generator(sampler, model, batch):
    with torch.no_grad():
        y_pred = model(batch[1], batch[2], batch[3]).squeeze(-1).cpu()
    return y_pred

def CrossUNet_loss_func(sampler, model, batch):
    loss = sampler.diffusion_loss_fn_unet(model, batch[0], batch[1], batch[2], batch[3])
    return loss

def CrossUNet_generator(sampler, model, batch):
    y_pred = sampler.ddim_sample_unet(model, batch[0].shape, batch[1], batch[2], batch[3]).cpu()
    return y_pred

def PRNet_loss_func(sampler, model, batch):
    guss_loss = nn.GaussianNLLLoss()
    treated_input, control_input, drug_input = batch
    b_size =  treated_input.shape[0]
    noise = torch.randn(b_size,10).to(treated_input.device)
    output = model(control_input, drug_input, noise)
    dim = output.size(1) // 2
    gene_means = output[:, :dim]
    gene_vars = output[:, dim:]
    gene_vars = F.softplus(gene_vars)
    loss = guss_loss(input=gene_means, target=treated_input, var=gene_vars)
    return loss

def PRNet_generator(sampler, model, batch):
    with torch.no_grad():
        treated_input, control_input, drug_input = batch
        b_size =  treated_input.shape[0]
        noise = torch.randn(b_size,10).to(treated_input.device)
        y_pred = model(control_input, drug_input, noise).cpu()
        # VAE sampling
        dim = y_pred.size(1) // 2
        gene_means = y_pred[:, :dim]
        gene_vars = y_pred[:, dim:]
        gene_vars = F.softplus(gene_vars)
        # reconstruction_loss = guss_loss(input=gene_means, target=treated_input, var=gene_vars)
        dist = normal.Normal(
            torch.clamp(
                torch.Tensor(gene_means),
                min=1e-3,
                max=1e3,
            ),
            torch.clamp(
                torch.Tensor(gene_vars.sqrt()),
                min=1e-3,
                max=1e3,
            )           
        )
        y_pred = dist.sample()
    return y_pred

def choose_loss_generator(config):
    model_type = config['model']['model_type']
    if model_type == "EDM":
        return EDM_loss_func, EDM_generator
    elif model_type == "EDMCross" or model_type == "EDMBasicCross":
        return EDM_Cross_loss_func, EDM_Cross_generator
    elif model_type == "Ada" or model_type == "Adawd":
        return Ada_loss_func, Ada_generator
    elif model_type == "DirectAda":
        return DirectAda_loss_func, DirectAda_generator
    elif model_type == "CatCross":
        return CatCross_loss_func, CatCross_generator
    elif model_type == "Cross":
        if 'using_recons' in config and config['using_recons']:
            print('Using recons loss and sampler')
            return Cross_loss_func_recons, Cross_generator_recons
        else:
            return Cross_loss_func, Cross_generator
    elif model_type == "exp2exp":
        return Cross_loss_func, Crossexp2exp_generator
    elif model_type == "DirectCross":
        return DirectCross_loss_func, DirectCross_generator
    elif model_type == "CrossU":
        return CrossUNet_loss_func, CrossUNet_generator
    elif model_type == "PR":
        return PRNet_loss_func, PRNet_generator
    else:
        raise NotImplementedError
