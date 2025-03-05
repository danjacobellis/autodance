import torch
import torch.nn as nn
import einops
import numpy as np
from PIL import Image
from evit import EfficientVitLargeStageND, GroupNorm8
from monarch import FactorizedConvND, FactorizedResBlockGNND
from tft.utils import compand, decompand
from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image
from torch.distributions.laplace import Laplace

class LaplaceCompand(nn.Module):
    def __init__(self, num_channels):
        super(LaplaceCompand, self).__init__()
        self.sigma = nn.Parameter(42.0*torch.ones(num_channels))

    def forward(self, x):
        shape = [1, -1] + [1] * (x.dim() - 2)
        sigma = self.sigma.view(shape).clamp(min=1e-6)
        laplace = Laplace(loc=0., scale=sigma)
        cdf = laplace.cdf(x)
        out = 254 * cdf - 127
        return out

class QuantizeLF8(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.compand = LaplaceCompand(num_channels)

    def forward(self, x):
        x = self.compand(x)
        if self.training:
            x += torch.rand_like(x) - 0.5
        return x

class AutoEncoderND(nn.Module):
    def __init__(self, dim, input_channels, J, latent_dim, lightweight_encode=True, lightweight_decode=False):
        super().__init__()
        assert dim in (1, 2, 3), "Dimension should be 1, 2 or 3."
        self.hidden_dim = input_channels * (2 ** (dim * J))
        self.dim = dim
        self.J = J
        self.latent_dim = latent_dim
        self.lightweight_encode = lightweight_encode
        self.lightweight_decode = lightweight_decode

        if dim == 1:
            from tft.wavelet import WPT1D, IWPT1D, DWT1DForward, DWT1DInverse
            self.wt = DWT1DForward(J=1, wave='bior4.4')
            self.wpt = WPT1D(wt=self.wt, J=self.J)
            self.iwt = DWT1DInverse(wave='bior4.4')
            self.iwpt = IWPT1D(iwt=self.iwt, J=self.J)
        elif dim == 2:
            from tft.wavelet import WPT2D, IWPT2D, DWT2DForward, DWT2DInverse
            self.wt = DWT2DForward(J=1, wave='bior4.4')
            self.wpt = WPT2D(wt=self.wt, J=self.J)
            self.iwt = DWT2DInverse(wave='bior4.4')
            self.iwpt = IWPT2D(iwt=self.iwt, J=self.J)
        elif dim == 3:
            from tft.wavelet import WPT3D, IWPT3D, DWT3DForward, DWT3DInverse
            self.wt = DWT3DForward(J=1, wave='bior4.4')
            self.wpt = WPT3D(wt=self.wt, J=self.J)
            self.iwt = DWT3DInverse(wave='bior4.4')
            self.iwpt = IWPT3D(iwt=self.iwt, J=self.J)

        if lightweight_encode:
            self.encoder_blocks = nn.Sequential(
                *[FactorizedResBlockGNND(dim, self.hidden_dim) for _ in range(6)]
            )
        else:
            self.encoder_blocks = EfficientVitLargeStageND(
                dim=dim,
                in_chs=self.hidden_dim,
                depth=6,
                norm_layer=GroupNorm8
            )

        self.conv_down = FactorizedConvND(dim, self.hidden_dim, latent_dim, bias=False)
        self.quantize = QuantizeLF8(latent_dim)
        self.conv_up = FactorizedConvND(dim, latent_dim, self.hidden_dim, kernel_size=1, bias=False)

        if lightweight_decode:
            self.decoder_blocks = nn.Sequential(
                *[FactorizedResBlockGNND(dim, self.hidden_dim) for _ in range(6)]
            )
        else:
            self.decoder_blocks = EfficientVitLargeStageND(
                dim=dim,
                in_chs=self.hidden_dim,
                depth=6,
                norm_layer=GroupNorm8
            )

    def encode(self, x):
        x = self.wpt(x)
        x = 12.8 * compand(x, power=0.4)
        x = self.encoder_blocks(x)
        x = self.conv_down(x)
        return x

    def decode(self, x):
        x = self.conv_up(x)
        x = self.decoder_blocks(x)
        x = decompand(x / 12.8, power=0.4)
        x = self.iwpt(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        rate = self.quantize.compand(x).std().log2()
        x = self.quantize(x)
        x = self.decode(x)
        return x, rate