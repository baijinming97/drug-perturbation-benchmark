import torch,math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange,repeat
# These classes are partially cited from https://github.com/YangLabHKUST/SpatialScope

# Conv1d with orthogonal initialization
class Conv1dWithInitialization(nn.Module):
    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)
        #torch.nn.init.constant_(self.conv1d.weight.data,3)

    def forward(self, x):
        return self.conv1d(x)


# Downsample and Upsample
class Downsample(nn.Module):
    def __init__(self, in_channels, scale_factor, with_conv = True):
        super().__init__()
        self.with_conv = with_conv
        
        
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=7,
                                        stride=scale_factor,
                                        padding=3)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, scale_factor, remain_dim=0, with_conv = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.ConvTranspose1d(in_channels,
                                                 in_channels,
                                                 kernel_size=7,
                                                 stride=scale_factor,
                                                 padding=3,
                                                 output_padding = remain_dim)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        return x
    

# Cblock
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ConvolutionBlock, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.convolution = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation
        )
    
    def forward(self, x):
        outputs = self.leaky_relu(x)
        outputs = self.convolution(outputs)
        return outputs

class CBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor, dilations, downsampling = True, remain_dim = None):
        super().__init__()
        if downsampling:
            sampling_layer1 = Downsample(in_channels = in_channels, scale_factor = factor)
            sampling_layer2 = Downsample(in_channels = out_channels, scale_factor = factor)
        else:
            sampling_layer1 = Upsample(in_channels = in_channels, scale_factor = factor, remain_dim = remain_dim)
            sampling_layer2 = Upsample(in_channels = out_channels, scale_factor = factor, remain_dim = remain_dim)          
        in_sizes = [in_channels] + [out_channels for _ in range(len(dilations) - 1)]
        out_sizes = [out_channels for _ in range(len(in_sizes))]
        self.main_branch = torch.nn.Sequential(*([
            sampling_layer1
        ] + [
            ConvolutionBlock(in_size, out_size, dilation)
            for in_size, out_size, dilation in zip(in_sizes, out_sizes, dilations)
        ]))
        self.residual_branch = torch.nn.Sequential(*[
            Conv1dWithInitialization(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
            sampling_layer2
        ])

    def forward(self, x):
        outputs = self.main_branch(x)
        outputs = outputs + self.residual_branch(x)
        return outputs
    

# FILM
class FILM(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels == None:
            out_channels = in_channels
        self.signal_conv = torch.nn.Sequential(*[
            Conv1dWithInitialization(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.LeakyReLU(0.2)
        ])
        self.scale_conv = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.shift_conv = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        outputs = self.signal_conv(x)
        scale, shift = self.scale_conv(outputs), self.shift_conv(outputs)
        return scale, shift
    

# Mblock
class FeatureWiseAffine(nn.Module):
    def __init__(self, n_channels):
        super(FeatureWiseAffine, self).__init__()
        self.instance_norm = torch.nn.InstanceNorm1d(n_channels)

    def forward(self, x, scale, shift):
        x = self.instance_norm(x)
        outputs = scale * x + shift
        return outputs
    
class BasicModulationBlock(nn.Module):
    """
    Linear modulation part of UBlock, represented by sequence of the following layers:
        - Feature-wise Affine
        - LReLU
        - 3x1 Conv
    """
    def __init__(self, n_channels, dilation):
        super(BasicModulationBlock, self).__init__()
        self.featurewise_affine = FeatureWiseAffine(n_channels)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.convolution = Conv1dWithInitialization(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation
        )

    def forward(self, x, scale, shift):
        outputs = self.featurewise_affine(x, scale, shift)
        outputs = self.leaky_relu(outputs)
        outputs = self.convolution(outputs)
        return outputs

class MBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor, dilations, downsampling = False, remain_dim = None):
        super().__init__()
        if downsampling:
            sampling_layer1 = Downsample(in_channels = in_channels, scale_factor = factor)
            sampling_layer2 = Downsample(in_channels = out_channels, scale_factor = factor)
        else:
            sampling_layer1 = Upsample(in_channels = in_channels, scale_factor = factor, remain_dim = remain_dim)
            sampling_layer2 = Upsample(in_channels = out_channels, scale_factor = factor, remain_dim = remain_dim)
        self.first_block_main_branch = torch.nn.ModuleDict({
            'upsampling': torch.nn.Sequential(*[               
                sampling_layer1,
                Conv1dWithInitialization(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilations[0],
                    dilation=dilations[0]
                )
            ]),
            'modulation': BasicModulationBlock(
                out_channels, dilation=dilations[1]
            )
        })
        self.first_block_residual_branch = torch.nn.Sequential(*[
            Conv1dWithInitialization(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
            sampling_layer2
        ])
        self.second_block_main_branch = torch.nn.ModuleDict({
            f'modulation_{idx}': BasicModulationBlock(
                out_channels, dilation=dilations[2 + idx]
            ) for idx in range(2)
        })

    def forward(self, x, scale, shift):
        # First upsampling residual block
        outputs = self.first_block_main_branch['upsampling'](x)
        outputs = self.first_block_main_branch['modulation'](outputs, scale, shift)
        outputs = outputs + self.first_block_residual_branch(x)

        # Second residual block
        residual = self.second_block_main_branch['modulation_0'](outputs, scale, shift)
        outputs = outputs + self.second_block_main_branch['modulation_1'](residual, scale, shift)
        return outputs
    

# Cross attention
class MultiHead_CrossAttention(nn.Module):
    def __init__(self, d_drug, d_exp, dim, d_v, num_head, d_o):
        '''
        Args:
            dim: dimension for each time step
            num_head:num head for multi-head self-attention
        '''
        super().__init__()
        self.d_drug=d_drug
        self.d_exp=d_exp
        self.dim=dim
        self.d_v=d_v
        self.num_head=num_head
        self.q=nn.Linear(d_exp, dim*num_head) # extend the dimension for later spliting
        self.k=nn.Linear(d_drug, dim*num_head)
        self.v=nn.Linear(d_drug, d_v*num_head)
        self.fc_o=nn.Linear(num_head*d_v, d_o)

 
    def forward(self, xd, exp):
        B, N_drug, _ = xd.shape
        _, N_exp, _ = exp.shape
        q = self.q(exp).reshape(B, N_exp, self.num_head, self.dim).permute(0, 2, 1, 3) #B,head,N,dim
        k = self.k(xd).reshape(B, N_drug, self.num_head, self.dim).permute(0, 2, 1, 3) #B,head,N,dim
        v = self.v(xd).reshape(B, N_drug, self.num_head, self.d_v).permute(0, 2, 1, 3) #B,head,N,dim
        att = q@k.transpose(-1, -2)/ math.sqrt(self.dim)
        #print(att)
        att = att.softmax(dim=3) # 将多个注意力矩阵合并为一个
        x = (att@v).transpose(1, 2)
        x=x.reshape(B, N_exp, self.num_head*self.d_v)
        x=self.fc_o(x)
        return x

class mask_cross_attn(nn.Module):
    def __init__(self,cond_d,x_d,dim,heads,d_out,dropout=0.):
        super().__init__()
        self.heads=heads
        self.to_q=nn.Linear(x_d,dim*heads)
        self.to_k=nn.Linear(cond_d,dim*heads)
        self.to_v=nn.Linear(cond_d,dim*heads)
        self.scale=dim**-0.5
        self.out_proj=nn.Sequential(
            nn.Linear(dim*heads,d_out),
            nn.Dropout(dropout)
        )
    
    #input batch_size，token numbers after padding, and real token number tensors(word number)
    def getmask(self,batch_size,max_token,each_token):
        mask=torch.zeros((batch_size,max_token))
        for i,idx in enumerate(each_token):
            mask[i, idx:max_token] = 1
        return mask.unsqueeze(1).bool()

    def forward(self,cond,x,mask=None):
        #x(B,x_n,x_dim), cond(B,cond_n,cond_dim)
        batch_size,cond_n,_=cond.shape
        q=self.to_q(x)
        k=self.to_k(cond)
        v=self.to_v(cond)
        q,k,v=map(lambda t:rearrange(t,'b n (h d) -> b h n d',h=self.heads),(q,k,v))
        sim=einsum('b h n d, b h m d -> b h n m',q,k)*self.scale
        if mask is not None:
            max_neg_value = -torch.finfo(sim.dtype).max
            batch_mask=self.getmask(batch_size,cond_n,mask)
            batch_mask = repeat(batch_mask, 'b n m -> b h n m', h=self.heads).to(sim.device)
            sim.masked_fill_(batch_mask, max_neg_value)
        attn=sim.softmax(dim=-1)
        #print(attn)
        out=einsum('b h n m, b h m d->b h n d',attn,v)
        out=rearrange(out,'b h n d -> b n (h d)')
        return self.out_proj(out)      

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        #print(x.shape)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)
    
class BasicTransformerBlock(nn.Module):
    def __init__(self, d_drug, d_exp, dim,num_head, d_o,gated_ff=True,dropout=0.):
        super().__init__()
        #self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(d_o, dropout=dropout, glu=gated_ff)
        self.attn2 = mask_cross_attn(cond_d=d_drug, x_d=d_exp, dim=dim, heads=num_head, d_out=d_o)
        #self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(d_exp)
        self.norm3 = nn.LayerNorm(d_o)

    def forward(self, xd, exp, mask):
        exp = exp.permute(0,2,1)
        #x = self.attn1(self.norm1(x)) + x
        exp = self.attn2(x=self.norm2(exp), cond=xd, mask=mask) + exp
        exp = self.ff(self.norm3(exp)) + exp
        return exp.permute(0,2,1)
    

# time embedding
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

# Dim calculator
def conv1d_out_dim(input,padding,dilation,kernel,stride):
    return int((input+2*padding-dilation*(kernel-1)-1)/stride+1)
def convtranspose1d_out_dim(input,padding,dilation,kernel,stride,output_padding=0):
    return (input-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1
def out_padding(input,target,stride):
    return target-convtranspose1d_out_dim(input,3,1,7,stride)


# Unet
class cross_attention_unet(nn.Module):
    def __init__(self, input_dim: int, channel_dim: list = [32,64,128,196], strides: list = [3,4,5,1], 
                drug_input_dim: int = 16, n_heads: int = 4,num_steps: int = 50, device = torch.device("cpu")):
        super().__init__()
        self.device=device
        # Parameters
        self.channel_dim = channel_dim
        self.strides = strides
        self.Cblock_dilations = [[1,2,4]]
        self.Mblock_dilations = [[1,2,1,2]]
        # Modules
        self.time_embedding = DiffusionEmbedding(num_steps)
        self.diffusion_projections = torch.nn.ModuleList([nn.Linear(128, channel_dims) for channel_dims in self.channel_dim])
        
        # left branch
        self.pre_conv = Conv1dWithInitialization(
            in_channels=1,
            out_channels=self.channel_dim[0],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pre_conv_cond = Conv1dWithInitialization(
            in_channels=1,
            out_channels=self.channel_dim[0],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.left_Cblocks = torch.nn.ModuleList([
            CBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations,
                downsampling=True
            ) for in_size, out_size, factor, dilations in zip(
                self.channel_dim,
                self.channel_dim[1:] + [self.channel_dim[-1]],
                self.strides,
                self.Cblock_dilations * 4
            )
        ])
        self.left_films = torch.nn.ModuleList([
            FILM(
                in_channels=in_size
            ) for in_size in self.channel_dim[1:] + [self.channel_dim[-1]]
        ])
        self.left_Mblocks = torch.nn.ModuleList([
            MBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations,
                downsampling=True
            ) for in_size, out_size, factor, dilations in zip(
                self.channel_dim,
                self.channel_dim[1:] + [self.channel_dim[-1]],
                self.strides,
                self.Mblock_dilations * 4
            )
        ])
        self.left_Crossattn = torch.nn.ModuleList([
            BasicTransformerBlock(d_drug=d_drugs, d_exp=d_exps, dim=dims, num_head=n_heads, d_o=d_os
            ) for d_drugs, d_exps, dims, d_os  in zip(
                [drug_input_dim]*4,
                self.channel_dim[1:] + [self.channel_dim[-1]],
                [item // 2 for item in self.channel_dim[1:] + [self.channel_dim[-1]]],
                self.channel_dim[1:] + [self.channel_dim[-1]]
            )
        ])
        self.left_res_branch = torch.nn.ModuleList([torch.nn.Sequential(*[
            Conv1dWithInitialization(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
            Downsample(in_channels = out_channels, scale_factor = factor)
        ]) for in_channels, out_channels, factor in zip(
            self.channel_dim,
            self.channel_dim[1:] + [self.channel_dim[-1]],
            self.strides
        )
        ])

        # right branch
        dims = [input_dim]
        output_paddings = []
        input = input_dim
        for factor in strides: 
            input = conv1d_out_dim(input,3,1,7,factor)
            dims.append(input)

        for i,factor in enumerate(strides[:-1][::-1]):
            output_paddings.append(out_padding(dims[3-i],dims[2-i],factor))
        
        self.output_paddings = output_paddings


        self.right_Cblocks = torch.nn.ModuleList([
            CBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations,
                downsampling=False,
                remain_dim=out_padding
            ) for in_size, out_size, factor, dilations,out_padding  in zip(
                (np.array(self.channel_dim[1:][::-1]) * 2),
                self.channel_dim[:-1][::-1],
                self.strides[:-1][::-1],
                self.Cblock_dilations * 3,
                output_paddings
            )
        ])
        self.right_films = torch.nn.ModuleList([
            FILM(
                in_channels=in_size
            ) for in_size in self.channel_dim[:-1][::-1]
        ])
        self.right_Mblocks = torch.nn.ModuleList([
            MBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations,
                downsampling=False,
                remain_dim=out_padding
            ) for in_size, out_size, factor, dilations,out_padding in zip(
                (np.array(self.channel_dim[1:][::-1]) * 2),
                self.channel_dim[:-1][::-1],
                self.strides[:-1][::-1],
                self.Mblock_dilations * 3,
                output_paddings
            )
        ])
        self.right_Crossattn = torch.nn.ModuleList([
            BasicTransformerBlock(d_drug=d_drugs, d_exp=d_exps, dim=dims, num_head=n_heads, d_o=d_os
            ) for d_drugs, d_exps, dims, d_os  in zip(
                [drug_input_dim]*3,
                self.channel_dim[:-1][::-1],
                [item//2 for item in self.channel_dim[:-1][::-1]],
                self.channel_dim[:-1][::-1]
            )
        ])
        self.right_res_branch = torch.nn.ModuleList([torch.nn.Sequential(*[
            Conv1dWithInitialization(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
            Upsample(in_channels = out_channels, scale_factor = factor,remain_dim=out_padding)
        ]) for in_channels, out_channels, factor, out_padding in zip(
            (np.array(self.channel_dim[1:][::-1]) * 2),
            self.channel_dim[:-1][::-1],
            self.strides[:-1][::-1],
            output_paddings
        )
        ])

        self.post_conv = Conv1dWithInitialization(
            in_channels=self.channel_dim[0],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.final_fully_connect=nn.Linear(input_dim,input_dim)
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, x_cond, time, x_drug, mask):
        #x, x_cond dim:(B,channel,genes),x_drug dim:(B,tokens,embdding_dim)
        time_embed = self.time_embedding(time).squeeze(1)
        time_embeddings = []
        for time_projector in self.diffusion_projections:
            time_embeddings.append(time_projector(time_embed))
        
        statistics = []
        x_cond_lefts, x_lefts = [], []
        left_cblock_outputs = self.pre_conv_cond(x_cond)
        for cblock, film in zip(self.left_Cblocks, self.left_films):
            left_cblock_outputs = cblock(left_cblock_outputs)
            scale, shift = film(x=left_cblock_outputs)
            statistics.append([scale, shift])
            x_cond_lefts.append(left_cblock_outputs)
        
        left_mblock_outputs = self.pre_conv(x)
        for i, (left_res,mblock,cross_attn) in enumerate(zip(self.left_res_branch, self.left_Mblocks, self.left_Crossattn)):
            step_embedding = time_embeddings[i]
            left_mblock_outputs_res = left_res(left_mblock_outputs)
            #print(left_mblock_outputs.shape)
            #print(time_embeddings[i].shape)
            left_mblock_outputs = left_mblock_outputs+step_embedding.unsqueeze(-1)
            scale, shift = statistics[i]
            left_mblock_outputs = mblock(x=left_mblock_outputs, scale=scale, shift=shift)
            #print(left_mblock_outputs.shape)
            #print(x_drug.shape)
            left_mblock_outputs = cross_attn(x_drug,left_mblock_outputs,mask)
            left_mblock_outputs = left_mblock_outputs + left_mblock_outputs_res
            x_lefts.append(left_mblock_outputs)
        
        _, _ = x_cond_lefts.pop(), x_lefts.pop()
        statistics = []
        right_cblock_outputs = left_cblock_outputs
        for cblock, film in zip(self.right_Cblocks, self.right_films):
            #print(right_cblock_outputs.shape)
            right_cblock_outputs = cblock(torch.cat([right_cblock_outputs, x_cond_lefts.pop()], dim=1))
            scale, shift = film(x=right_cblock_outputs)
            statistics.append([scale, shift])
        
        right_mblock_outputs = left_mblock_outputs
        for i, (right_res,mblock,cross_attn) in enumerate(zip(self.right_res_branch, self.right_Mblocks, self.right_Crossattn)):
            step_embedding = time_embeddings[3-i]
            x_left_concat = x_lefts.pop()
            right_mblock_outputs_res = right_res(torch.cat([right_mblock_outputs, x_left_concat], dim=1))
            #print(right_mblock_outputs.shape,step_embedding.unsqueeze(-1).shape)
            right_mblock_outputs = right_mblock_outputs+step_embedding.unsqueeze(-1)
            scale, shift = statistics[i]
            right_mblock_outputs = mblock(x=torch.cat([right_mblock_outputs, x_left_concat], dim=1), scale=scale, shift=shift)
            right_mblock_outputs = cross_attn(x_drug,right_mblock_outputs,mask)
            right_mblock_outputs = right_mblock_outputs + right_mblock_outputs_res                
        outputs = self.post_conv(right_mblock_outputs)
        return(outputs)
        # outputs = self.final_fully_connect(outputs)
        # return(outputs.unsqueeze(1))

    

    
