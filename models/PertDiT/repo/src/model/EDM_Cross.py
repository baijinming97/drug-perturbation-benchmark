import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
from utils.seed_everything import seed_everything
from model.common import *

class BasicTransformer_block(nn.Module):
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
        self.mhca = MultiheadAttention(
            d_model, 
            num_heads = num_heads, 
            dropout = dropout,
            batch_first = True
        )
        self.ffn = GatedMLP(d_model, mlp_hidden_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)


    def forward(
        self, 
        x: torch.Tensor, 
        x_pre: torch.Tensor,
        x_drug: torch.Tensor,
        mask: torch.Tensor
        ):

        # add x_pre using cross attention
        x_pre = self.norm1(x_pre)
        x = self.mha(key = x_pre, query = self.norm2(x), value = x_pre, need_weights = False)[0] + x

        # add x_drug using cross attention
        x_drug = self.norm4(x_drug)
        x = self.mhca(key = x_drug, query = self.norm3(x), value = x_drug, key_padding_mask = mask, need_weights = False)[0] + x

        x = self.ffn(self.norm5(x))+x

        return x

class EDM_Crossformer(nn.Module):
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
                BasicTransformer_block(
                d_model,
                mlp_hidden_dim,
                num_heads,
                dropout,
            ) for _ in range(num_layers)
        ])
        self.sigma_encoder = EDM_PositionalEmbedding(embedding_dim = d_model)
        self.xpre_encoder = nn.Linear(1, d_model)
        self.xdrug_encoder = nn.Linear(d_cond, d_model)
        self.pre_proj = nn.Linear(1, d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, x_pre, x_drug, sigma, mask):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        x_pre = x_pre.to(torch.float32)
        x_drug = x_drug.to(torch.float32)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self._forward(c_in * x, x_pre, x_drug, c_noise.flatten(), mask)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x
    
    def _forward(self, x, x_pre, x_drug, sigma, mask):
        B, L, _ = x_drug.shape
        mask = getmask(B, L, mask).to(x_drug.device)
        sigma_embedding = self.sigma_encoder(sigma).unsqueeze(1)
        x = self.pre_proj(x.unsqueeze(-1))
        x_pre = self.xpre_encoder(x_pre.unsqueeze(-1))
        x_drug = self.xdrug_encoder(x_drug)
        # x = x + sigma_embedding + self.positional_encoding.unsqueeze(0)
        x = x + self.positional_encoding.unsqueeze(0)
        x_pre = x_pre + sigma_embedding + self.positional_encoding.unsqueeze(0)
        x_drug = x_drug + sigma_embedding
        for block in self.blocks:
            x = block(x,x_pre,x_drug,mask)
        x = self.out_proj(x).squeeze(-1)
        return x