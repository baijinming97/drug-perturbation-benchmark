import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from utils.seed_everything import seed_everything
from model.common import *

def modulate(x,shift,scale):
    return x*(1+scale.unsqueeze(1))+shift.unsqueeze(1)

class PreAdanorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
    
    def forward(self, x, shift, scale):
        norm_x = modulate(self.norm(x), shift, scale)
        return x, norm_x

class Adaformer_block(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        ):
        super().__init__()
        self.mha = MultiheadAttention(
            d_model, 
            num_heads = num_heads, 
            dropout = dropout,
            batch_first = True
        )
        self.ffn = GatedMLP(d_model, mlp_hidden_dim)
        self.norm1 = PreAdanorm(d_model)
        self.norm2 = PreAdanorm(d_model)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6*d_model, bias = True),
        )

    def forward(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor
        ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        x0, x = self.norm1(x, shift_msa, scale_msa)
        x = self.mha(key=x, query=x, value=x, need_weights = False)[0]
        x = x0 + gate_msa.unsqueeze(1) * x
        x0, x = self.norm2(x, shift_mlp, scale_mlp)
        x = self.ffn(x)
        x = x0 + gate_mlp.unsqueeze(1) * x
        return x

class EDM_Adaformer(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 32,
        d_cond: int = 1024,
        d_pre: int = 978,
        mlp_hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
        sigma_data = 0.5
        ):
        super().__init__()
        seed_everything(117)
        self.sigma_data = sigma_data
        self.positional_encoding = nn.Parameter(torch.randn(d_pre, d_model))
        self.blocks = nn.ModuleList([
                Adaformer_block(
                d_model,
                mlp_hidden_dim,
                num_heads,
                dropout,
            ) for _ in range(num_layers)
        ])
        self.sigma_encoder = EDM_PositionalEmbedding(embedding_dim = d_model)
        self.cond_encoder = nn.Linear(d_cond, d_model)
        self.pre_proj = nn.Linear(2, d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x, x_cond, x_drug, sigma):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        x_cond = x_cond.to(torch.float32)
        x_drug = x_drug.to(torch.float32)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        x_input = torch.stack([c_in * x, x_cond],dim=-1)
        F_x = self._forward(x_input, x_drug, c_noise.flatten())
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def _forward(self, x, cond, sigma):
        sigma_embedding = self.sigma_encoder(sigma)
        x = self.pre_proj(x)
        cond = self.cond_encoder(cond)
        x = x + sigma_embedding.unsqueeze(1) + self.positional_encoding.unsqueeze(0)
        cond = cond + sigma_embedding
        for block in self.blocks:
            x = block(x,cond)
        x = self.out_proj(x).squeeze(-1)
        return x
