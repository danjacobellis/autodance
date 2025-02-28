import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormAct3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1,
                 bias=False, norm_layer=nn.BatchNorm3d, act_layer=nn.GELU):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=bias)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = act_layer() if act_layer else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
        
class MBConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expand_ratio=6,
                 norm_layer=nn.BatchNorm3d, act_layer=nn.GELU):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.inverted_conv = ConvNormAct3D(in_channels, mid_channels, 1, norm_layer=None, act_layer=act_layer, bias=True)
        self.depth_conv = ConvNormAct3D(mid_channels, mid_channels, kernel_size, groups=mid_channels, norm_layer=None, act_layer=act_layer, bias=True)
        self.point_conv = ConvNormAct3D(mid_channels, out_channels, 1, norm_layer=norm_layer, act_layer=None, bias=False)

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class ResidualBlock3D(nn.Module):
    def __init__(self, main: nn.Module):
        super().__init__()
        self.main = main

    def forward(self, x):
        return x + self.main(x)

class LiteMLA3D(nn.Module):
    def __init__(self, in_channels, dim=32, norm_layer=nn.BatchNorm3d, kernel_func=nn.ReLU, scales=(3,), eps=1e-5):
        super().__init__()
        self.eps = eps
        heads = in_channels // dim
        total_dim = heads * dim
        self.dim = dim
        self.qkv = nn.Conv3d(in_channels, 3 * total_dim, kernel_size=1, bias=False)
        self.kernel_func = kernel_func()
        self.proj = ConvNormAct3D(total_dim, in_channels, kernel_size=1, norm_layer=norm_layer, act_layer=None)

    def forward(self, x):
        B, C, D, H, W = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, D*H*W).permute(0, 3, 1, 2)
        q, k, v = qkv.unbind(-1)
    
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        v = F.pad(v, (0, 1), mode='constant', value=1.)
    
        kv = torch.einsum('bnc,bnv->bcv', k, v)
        out = torch.einsum('bnc,bcv->bnv', q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)
    
        out = out.permute(0, 2, 1).reshape(B, -1, D, H, W)
        return self.proj(out)


class EfficientVitBlock3D(nn.Module):
    def __init__(self, in_channels, head_dim=32, expand_ratio=6, norm_layer=nn.BatchNorm3d, act_layer=nn.GELU):
        super().__init__()
        self.context_module = ResidualBlock3D(LiteMLA3D(in_channels, dim=head_dim, norm_layer=norm_layer))
        self.local_module = ResidualBlock3D(MBConv3D(in_channels, in_channels, expand_ratio=expand_ratio,
                                                     norm_layer=norm_layer, act_layer=act_layer))

    def forward(self, x):
        x = self.context_module(x)
        x = self.local_module(x)
        return x

class EfficientVitLargeStage3D(nn.Module):
    def __init__(self, in_chs: int, depth: int, norm_layer=nn.BatchNorm3d, act_layer=nn.GELU, head_dim: int = 32):
        super().__init__()
        self.blocks = nn.Sequential(*[EfficientVitBlock3D(in_chs, head_dim, norm_layer=norm_layer, act_layer=act_layer) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)