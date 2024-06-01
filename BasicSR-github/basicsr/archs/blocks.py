import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from einops import rearrange
from torch import einsum
import pdb

def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', skip=True, bias=True):
        super(ConvBlock, self).__init__()
        
        #pdb.set_trace()
        
        stride = 1 if scale == 'none' else 2
          
        if scale == 'up' or scale == 'down':
          self.skip = False
        else:
          self.skip = skip
        
        if scale == 'up':
          #self.reflection_pad = nn.ReflectionPad2d((kernel_size // 2)) 
          self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size*2, stride, padding = 2, bias=bias)
        else:
          #self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2) 
          self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = kernel_size // 2, bias=bias)
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        #self.reflection_pad_2 = nn.ReflectionPad2d(kernel_size // 2) 
        self.conv2d_2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride = 1, padding = kernel_size // 2, bias=bias)

    def forward(self, x):
        #pdb.set_trace()
        #out = self.reflection_pad(x)
        out = self.conv2d(x)
        out = self.relu(out)
        #out = self.reflection_pad_2(out)
        out = self.conv2d_2(out)
        if self.skip:
          return out + x
        else:
          return out

class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    ------------------
    # Args
        - hg_depth: depth of HourGlassBlock. 0: don't use attention map.
        - use_pmask: whether use previous mask as HourGlassBlock input.
    """
    def __init__(self, c_in, c_out, kernel_size=3, scale='none', bias=True):
        super(ResidualBlock, self).__init__()
        #pdb.set_trace()
        self.c_in = c_in
        self.c_out = c_out

        self.conv1 = ConvBlock(c_in, c_out, kernel_size, scale, bias)
        self.conv2 = ConvBlock(c_in, c_out, kernel_size, scale, bias)

        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, bn=False, act=nn.LeakyReLU(0.2, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class MeanShift(nn.Conv2d):
    def __init__(self,
                 rgb_range,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0),
                 sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x_q, x_kv):
        _, _, dim, heads = *x_q.shape, self.heads
        _, _, dim_large = x_kv.shape 

        assert dim == dim_large

        q = self.to_q(x_q)

        q = rearrange(q, 'b n (h d) -> b h n d', h=heads)

        kv = self.to_kv(x_kv).chunk(2, dim=-1)
        
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), kv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm2(x2), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        #pdb.set_trace()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)