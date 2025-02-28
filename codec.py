import torch
import torch.nn as nn
from evit import EfficientVitLargeStageND, GroupNorm8
from monarch import FactorizedConvND, FactorizedResBlockGNND
from tft.utils import compand, decompand

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

class AutoEncoderND(nn.Module):
    def __init__(self, dim, input_channels, J, latent_dim, num_res_blocks=6):
        super().__init__()
        assert dim in (1, 2, 3), "Dimension should be 1, 2 or 3."
        self.hidden_dim = input_channels*(2**(dim*J))
        self.dim = dim
        self.J = J
        self.latent_dim = latent_dim
        
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

        self.resblocks = nn.Sequential(
            *[FactorizedResBlockGNND(dim, self.hidden_dim) for _ in range(num_res_blocks)]
        )
        self.conv_down = FactorizedConvND(dim, self.hidden_dim, latent_dim, bias=False)
        self.quantize = Quantize8bits()
        self.conv_up = FactorizedConvND(dim, latent_dim, self.hidden_dim, kernel_size=1, bias=False)
        self.vit = EfficientVitLargeStageND(
            dim=dim,
            in_chs=self.hidden_dim,
            depth=6,
            norm_layer=GroupNorm8
        )

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