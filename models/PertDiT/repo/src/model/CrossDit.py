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

class BasicCatTransformer_block(nn.Module):
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
        # add x_pre using cross attention
        x1 = self.norm1(x)
        x = self.mha(key = x1, query = x1, value = x1, need_weights = False)[0] + x

        # add x_drug using cross attention
        x_drug = self.norm2(x_drug)
        x = self.mhca(key = x_drug, query = self.norm3(x), value = x_drug, key_padding_mask = mask, need_weights = False)[0] + x
        x = self.ffn(self.norm4(x))+x

        return x

class BasicTransformer_block_nonorm(nn.Module):
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
        x_pre: torch.Tensor,
        x_drug: torch.Tensor,
        mask: torch.Tensor
        ):

        # add x_pre using cross attention
        x = self.mha(key = x_pre, query = self.norm1(x), value = x_pre, need_weights = False)[0] + x

        # add x_drug using cross attention
        x_drug = self.norm2(x_drug)
        x = self.mhca(key = x_drug, query = self.norm3(x), value = x_drug, key_padding_mask = mask, need_weights = False)[0] + x
        x = self.ffn(self.norm4(x))+x

        return x
    
class SimpleCross_block(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        ):
        super().__init__()
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

    def forward(
        self, 
        x: torch.Tensor, 
        x_drug: torch.Tensor,
        mask: torch.Tensor
        ):
        # add x_drug using cross attention
        x_drug = self.norm1(x_drug)
        x = self.mhca(key = x_drug, query = self.norm2(x), value = x_drug, key_padding_mask = mask, need_weights = False)[0] + x
        x = self.ffn(self.norm3(x))+x

        return x
    
class Crossformer(nn.Module):
    def __init__(
        self,
        num_steps: int = 50,
        num_layers: int = 2,
        d_model: int = 32,
        d_cond: int = 1024,
        d_pre: int = 978,
        mlp_hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
        nonorm: bool = False
        ):
        super().__init__()
        seed_everything(117)
        self.positional_encoding = nn.Parameter(torch.randn(d_pre, d_model))
        if not nonorm:
            self.blocks = nn.ModuleList([
                    BasicTransformer_block(
                    d_model,
                    mlp_hidden_dim,
                    num_heads,
                    dropout,
                ) for _ in range(num_layers)
            ])
        else:
            print('remove pre_x norm')
            self.blocks = nn.ModuleList([
                    BasicTransformer_block_nonorm(
                    d_model,
                    mlp_hidden_dim,
                    num_heads,
                    dropout,
                ) for _ in range(num_layers)
            ])
        self.time_encoder = DiffusionEmbedding(num_steps, embedding_dim = 2*d_model, projection_dim = d_model)
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

    def forward(self, x, x_pre, x_drug, time, mask):
        B, L, _ = x_drug.shape
        mask = getmask(B, L, mask).to(x_drug.device)
        time_embedding = self.time_encoder(time)
        x = self.pre_proj(x.unsqueeze(-1))
        x_pre = self.xpre_encoder(x_pre.unsqueeze(-1))
        x_drug = self.xdrug_encoder(x_drug)
        x = x + time_embedding + self.positional_encoding.unsqueeze(0)
        x_pre = x_pre + time_embedding + self.positional_encoding.unsqueeze(0)
        x_drug = x_drug + time_embedding
        for block in self.blocks:
            x = block(x,x_pre,x_drug,mask)
        x = self.out_proj(x)
        return x
    
class CatCrossformer(nn.Module):
    def __init__(
        self,
        num_steps: int = 50,
        num_layers: int = 2,
        d_model: int = 32,
        d_cond: int = 1024,
        d_pre: int = 978,
        mlp_hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
        block_type: str = "simple",
        ):
        super().__init__()
        seed_everything(117)
        self.positional_encoding = nn.Parameter(torch.randn(d_pre, d_model))
        if block_type=="simple":
            self.blocks = nn.ModuleList([
                    SimpleCross_block(
                    d_model,
                    mlp_hidden_dim,
                    num_heads,
                    dropout,
                ) for _ in range(num_layers)
            ])
        elif block_type=="basic":
            self.blocks = nn.ModuleList([
                    BasicCatTransformer_block(
                    d_model,
                    mlp_hidden_dim,
                    num_heads,
                    dropout,
                ) for _ in range(num_layers)
            ])
        else:
            raise NotImplementedError
        self.time_encoder = DiffusionEmbedding(num_steps, embedding_dim = 2*d_model, projection_dim = d_model)
        self.xdrug_encoder = nn.Linear(d_cond, d_model)
        self.pre_proj = nn.Linear(2, d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, x_drug, time, mask):
        B, L, _ = x_drug.shape
        mask = getmask(B, L, mask).to(x_drug.device)
        time_embedding = self.time_encoder(time)
        x = self.pre_proj(x)
        x_drug = self.xdrug_encoder(x_drug)
        x = x + time_embedding + self.positional_encoding.unsqueeze(0)
        x_drug = x_drug + time_embedding
        for block in self.blocks:
            x = block(x,x_drug,mask)
        x = self.out_proj(x)
        return x