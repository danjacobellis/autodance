import io
import torch
import torch.nn as nn
import PIL.Image
import einops
from timm.models.efficientvit_mit import GELUTanh, EfficientVitBlock, ResidualBlock, build_local_block
from tft.wavelet import WPT2D, IWPT2D, DWT2DForward, DWT2DInverse
from tft.utils import compand, decompand
from torchvision.transforms.v2 import ToPILImage, PILToTensor

class FactorizedConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, bias=False, kernel_size=1):
        super().__init__()
        self.in_chs, self.out_chs = in_chs, out_chs
        g1, g2 = self._pick_groups(in_chs, out_chs)
        self.g1, self.g2 = g1, g2
        self.conv1 = nn.Conv2d(
            in_chs, 
            in_chs, 
            kernel_size=kernel_size, 
            groups=g1, 
            bias=bias, 
            padding=(kernel_size - 1) // 2
        )
        self.conv2 = nn.Conv2d(
            in_chs, 
            out_chs, 
            kernel_size=kernel_size, 
            groups=g2, 
            bias=bias, 
            padding=(kernel_size - 1) // 2
        )

    def _pick_groups(self, in_chs, out_chs):
        def _divisors(n):
            divs = []
            i = 1
            while i * i <= n:
                if n % i == 0:
                    divs.append(i)
                    if i != n // i:
                        divs.append(n // i)
                i += 1
            return sorted(divs)
        divs = _divisors(in_chs)
        best_g1, best_g2, best_cost = 1, in_chs, float('inf')
        for d in divs:
            g1, g2 = d, in_chs // d
            cost = (in_chs * in_chs / g1) + (in_chs * out_chs / g2)
            if cost < best_cost:
                best_cost, best_g1, best_g2 = cost, g1, g2
        return best_g1, best_g2

    def forward(self, x):
        x = self.conv1(x)
        x = einops.rearrange(
            x, 
            'b (g1 g2) h w -> b (g2 g1) h w', 
            g1=self.g1, 
            g2=self.g2
        )
        x = self.conv2(x)
        return x

class FactorizedSqueezeExcite(nn.Module):
    def __init__(self, channels, se_ratio=0.25):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid_chs = int(channels * se_ratio)
        self.conv_reduce = FactorizedConv2d(
            in_chs=channels,
            out_chs=mid_chs,
            kernel_size=1,
            bias=True
        )
        self.act = GELUTanh()
        self.conv_expand = FactorizedConv2d(
            in_chs=mid_chs,
            out_chs=channels,
            kernel_size=1,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.conv_reduce(s)
        s = self.act(s)
        s = self.conv_expand(s)
        s = self.sigmoid(s)
        return x * s

class GroupNorm8(nn.Module):
    def __init__(self, num_features, eps=1e-7, affine=True):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=8,
                                      num_channels=num_features,
                                      eps=eps,
                                      affine=affine)
    def forward(self, x):
        return self.groupnorm(x)

class FactorizedResBlockGN(nn.Module):
    def __init__(self, channels, se_ratio=0.25):
        super().__init__()
        self.conv1 = FactorizedConv2d(channels, channels, kernel_size=3)
        self.gn1 = GroupNorm8(channels)
        self.act1 = GELUTanh()
        self.conv2 = FactorizedConv2d(channels, channels, kernel_size=3)
        self.gn2 = GroupNorm8(channels)
        self.se = FactorizedSqueezeExcite(channels, se_ratio=se_ratio)

    def forward(self, x):
        skip = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.se(out)
        out = out + skip
        return out

class Quantize8bits(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardtanh = nn.Hardtanh(min_val=-126.99, max_val=126.99)

    def forward(self, x):
        if self.training:
            x = self.hardtanh(x)
            x = x + torch.rand_like(x) - 0.5
            return x
        else:
            x = self.hardtanh(x)
            x = torch.round(x)
            return x

class EfficientVitStageNoDS(nn.Module):
    def __init__(self, in_chs, out_chs, depth, norm_layer, act_layer):
        super().__init__()
        blocks = [
            ResidualBlock(
                build_local_block(
                    in_channels=in_chs,
                    out_channels=out_chs,
                    stride=1,
                    expand_ratio=4,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    fewer_norm=True,
                    block_type='default'
                ),
                None,
            )
        ]
        in_chs = out_chs
        for _ in range(depth):
            blocks.append(
                EfficientVitBlock(
                    in_channels=in_chs,
                    head_dim=32,
                    expand_ratio=4,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

def entropy_code(ints):
    webp_bytes = []
    for sample in ints:
        c_groups = sample.shape[0] // 3
        sample_webp = []
        for g in range(c_groups):
            group = sample[g*3 : g*3+3]
            group = (group + 127).clamp_(0, 255).byte()
            img = ToPILImage()(group)
            buff = io.BytesIO()
            img.save(buff, format='WEBP', lossless=True)
            sample_webp.append(buff.getbuffer())
        webp_bytes.append(sample_webp)
    return webp_bytes

def entropy_decode(webp_bytes):
    batch_out = []
    for sample_buffers in webp_bytes:
        group_tensors = []
        for buff in sample_buffers:
            with io.BytesIO(buff) as memfile:
                img = PIL.Image.open(memfile).convert('RGB')
            t = PILToTensor()(img).to(torch.int16) - 127
            group_tensors.append(t)
        sample_tensor = torch.cat(group_tensors, dim=0)
        sample_tensor = sample_tensor.unsqueeze(0)
        batch_out.append(sample_tensor)
    decoded = torch.cat(batch_out, dim=0)
    return decoded

class AutoEncoder2D(nn.Module):
    def __init__(self, input_channels, J, latent_dim, num_res_blocks=6):
        super().__init__()
        self.hidden_dim = input_channels * (4 ** J)
        self.J = J
        self.latent_dim = latent_dim
        self.wt = DWT2DForward(J=1, wave='bior4.4')
        self.wpt = WPT2D(wt=self.wt, J=self.J)
        self.resblocks = nn.Sequential(
            *[FactorizedResBlockGN(self.hidden_dim) 
              for _ in range(num_res_blocks)]
        )
        self.conv_down = FactorizedConv2d(self.hidden_dim, latent_dim, bias=False)
        self.quantize = Quantize8bits()
        self.conv_up = nn.Conv2d(latent_dim, self.hidden_dim, kernel_size=1, padding=0, bias=False)
        self.vit = EfficientVitStageNoDS(
            in_chs=self.hidden_dim,
            out_chs=self.hidden_dim,
            depth=6,
            norm_layer=GroupNorm8,
            act_layer=GELUTanh,
        )
        self.iwt = DWT2DInverse(wave='bior4.4')
        self.iwpt = IWPT2D(iwt=self.iwt, J=self.J)

    def encode(self, x):
        x = self.wpt(x)
        x = 12.8 * compand(x, power=0.4)
        x = self.resblocks(x)
        x = self.conv_down(x)
        return x

    def decode(self, x):
        x = self.conv_up(x)
        x = self.vit(x)
        x = decompand(x / 12.8, power=0.4)
        x = self.iwpt(x)
        return x
        
    def forward(self, x):
        x = self.encode(x)
        rate = self.quantize.hardtanh(x).std().log2()
        x = self.quantize(x)
        x = self.decode(x)
        return x, rate