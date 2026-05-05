import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

class PRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.PAdapter = nn.Sequential(
            nn.Linear(1024,128,bias=False),
            nn.Linear(128,64,bias=True)
        )
        self.PEncoder = nn.Sequential(
            nn.Linear(1042,128,bias=False),
            nn.Linear(128,64,bias=True)
        )
        self.PDecoder = nn.Sequential(
            nn.Linear(in_features=128, out_features=128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Dropout(p=0.05),
            nn.Linear(in_features=128, out_features=978*2, bias=True)
        )
        self.PDecoder_noise = nn.Sequential(
            nn.Linear(in_features=138, out_features=128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Dropout(p=0.05),
            nn.Linear(in_features=128, out_features=978*2, bias=True)
        )
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                module.reset_parameters()

    def forward_vae(self, x: torch.Tensor, c: torch.Tensor, n: torch.Tensor):
        c = self.PAdapter(c)
        noise = (x, c)
        input_ = torch.cat(noise, 1)
        z = self.PEncoder(input_)
        z_c = (z, c, n)
        z_c_ = torch.cat(z_c, 1)
        x_hat = self.PDecoder_noise(z_c_)
        return x_hat

    def forward_ae(self, x: torch.Tensor, c: torch.Tensor):
        c = self.PAdapter(c)
        noise = (x, c)
        input_ = torch.cat(noise, 1)
        z = self.PEncoder(input_)
        z_c = (z, c)
        z_c_ = torch.cat(z_c, 1)
        x_hat = self.PDecoder(z_c_)
        return x_hat

    def forward(self, x: torch.Tensor, c: torch.Tensor, n=None):
        if n is None:
            return self.forward_ae(x,c)
        else:
            return self.forward_vae(x,c,n)