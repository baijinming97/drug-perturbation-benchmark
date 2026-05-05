import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.silu,
        bias=True,
        multiple_of=32,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class EDM_PositionalEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=128, projection_dim=None, max_positions=10000, endpoint=False):
        super().__init__()
        if projection_dim is None:
            projection_dim = 4*embedding_dim
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, embedding_dim)

        freqs = torch.arange(start=0, end=self.embedding_dim//2, dtype=torch.float32)
        freqs = freqs / (self.embedding_dim // 2 - (1 if self.endpoint else 0))
        self.freqs = (1 / self.max_positions) ** freqs

    def forward(self, x):
        freqs = self.freqs.to(x.device)
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

def getmask(batch_size,max_token,each_token):
    mask=torch.zeros((batch_size,max_token),dtype=bool)
    for i,idx in enumerate(each_token):
        mask[i, idx:] = True
    return mask