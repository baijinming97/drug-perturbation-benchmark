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


    def forward(
        self, 
        x: torch.Tensor, 
        x_drug: torch.Tensor,
        mask: torch.Tensor
        ):

        # add x_pre using adaptive layernorm
        x = self.norm1(x)
        x = self.mha(key = x, query = x, value = x, need_weights = False)[0] + x

        # add x_drug using cross attention
        x_drug = self.norm2(x_drug)
        x = self.mhca(key = x_drug, query = self.norm3(x), value = x_drug, key_padding_mask = mask, need_weights = False)[0] + x

        x = self.ffn(self.norm4(x))+x

        return x
        
class DirectCrossformer(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        d_model: int = 32,
        d_cond: int = 1024,
        d_pre: int = 978,
        mlp_hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
        ):
        super().__init__()
        seed_everything(117)
        self.positional_encoding = nn.Parameter(torch.randn(d_pre, d_model))
        self.blocks = nn.ModuleList([
                BasicTransformer_block(
                d_model,
                mlp_hidden_dim,
                num_heads,
                dropout,
            ) for _ in range(num_layers)
        ])
        self.xdrug_encoder = nn.Linear(d_cond, d_model)
        self.pre_proj = nn.Linear(1, d_model)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, 1),
            # nn.ReLU()
        )
            
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x, x_drug, mask):
        B, L, _ = x_drug.shape
        mask = getmask(B, L, mask).to(x_drug.device)
        x = self.pre_proj(x.unsqueeze(-1))
        x_drug = self.xdrug_encoder(x_drug)
        x = x + self.positional_encoding.unsqueeze(0)
        for block in self.blocks:
            x = block(x,x_drug,mask)
        x = self.out_proj(x)
        return x