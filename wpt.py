# Modified from github.com/fbcotter/pytorch_wavelets/blob/master/pytorch_wavelets/dwt/lowlevel.py
# Changes:
# Removed all modes other than periodic
# Added Support for 3D transforms
# Added 1D, 2D, and 3D wavelet packet transform

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import einops
from torch.autograd import Function

def roll(x, n, dim, make_even=False):
    if n < 0:
        n = x.shape[dim] + n
    if make_even and x.shape[dim] % 2 == 1:
        end = 1
    else:
        end = 0
    if dim == 0:
        return torch.cat((x[-n:], x[:-n+end]), dim=0)
    elif dim == 1:
        return torch.cat((x[:,-n:], x[:,:-n+end]), dim=1)
    elif dim == 2:
        return torch.cat((x[:,:,-n:], x[:,:,:-n+end]), dim=2)
    elif dim == 3:
        return torch.cat((x[:,:,:,-n:], x[:,:,:,:-n+end]), dim=3)
    elif dim == 4:
        return torch.cat((x[:,:,:,:,-n:], x[:,:,:,:,:-n+end]), dim=4)
    else:
        raise ValueError("Dimension out of range")

def prep_filt_afb1d(h0, h1, device=None):
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
    return h0, h1

def afb1d(x, h0, h1, dim=-1):
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1,1,1,1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    if x.shape[dim] % 2 == 1:
        if d == 2:
            x = torch.cat((x, x[:,:,-1:]), dim=2)
        else:
            x = torch.cat((x, x[:,:,:,-1:]), dim=3)
        N += 1
    x = roll(x, -L2, dim=d)
    pad = (L-1, 0) if d == 2 else (0, L-1)
    lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
    N2 = N//2
    if d == 2:
        lohi[:,:,:L2] = lohi[:,:,:L2] + lohi[:,:,N2:N2+L2]
        lohi = lohi[:,:,:N2]
    else:
        lohi[:,:,:,:L2] = lohi[:,:,:,:L2] + lohi[:,:,:,N2:N2+L2]
        lohi = lohi[:,:,:,:N2]

    return lohi
        
class AFB1D(Function):
    @staticmethod
    def forward(ctx, x, h0, h1):

        # Make inputs 4d
        x = x[:, :, None, :]
        h0 = h0[:, :, None, :]
        h1 = h1[:, :, None, :]

        # Save for backwards
        ctx.save_for_backward(h0, h1)
        ctx.shape = x.shape[3]

        lohi = afb1d(x, h0, h1, dim=3)
        x0 = lohi[:, ::2, 0].contiguous()
        x1 = lohi[:, 1::2, 0].contiguous()
        return x0, x1

    @staticmethod
    def backward(ctx, dx0, dx1):
        dx = None
        if ctx.needs_input_grad[0]:
            h0, h1 = ctx.saved_tensors

            # Make grads 4d
            dx0 = dx0[:, :, None, :]
            dx1 = dx1[:, :, None, :]

            dx = sfb1d(dx0, dx1, h0, h1, dim=3)[:, :, 0]

            # Check for odd input
            if dx.shape[2] > ctx.shape:
                dx = dx[:, :, :ctx.shape]

        return dx, None, None, None, None, None

def prep_filt_sfb1d(g0, g1, device=None):
    g0 = np.array(g0).ravel()
    g1 = np.array(g1).ravel()
    t = torch.get_default_dtype()
    g0 = torch.tensor(g0, device=device, dtype=t).reshape((1, 1, -1))
    g1 = torch.tensor(g1, device=device, dtype=t).reshape((1, 1, -1))

    return g0, g1

def sfb1d(lo, hi, g0, g1, dim=-1):
    C = lo.shape[1]
    d = dim % 4
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    N = 2*lo.shape[d]
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1,2)
    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)
    y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
        F.conv_transpose2d(hi, g1, stride=s, groups=C)
    if d == 2:
        y[:,:,:L-2] = y[:,:,:L-2] + y[:,:,N:N+L-2]
        y = y[:,:,:N]
    else:
        y[:,:,:,:L-2] = y[:,:,:,:L-2] + y[:,:,:,N:N+L-2]
        y = y[:,:,:,:N]
    y = roll(y, 1-L//2, dim=dim)

    return y

class SFB1D(Function):
    @staticmethod
    def forward(ctx, low, high, g0, g1):
        # Make into a 2d tensor with 1 row
        low = low[:, :, None, :]
        high = high[:, :, None, :]
        g0 = g0[:, :, None, :]
        g1 = g1[:, :, None, :]

        ctx.save_for_backward(g0, g1)

        return sfb1d(low, high, g0, g1, dim=3)[:, :, 0]

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            g0, g1, = ctx.saved_tensors
            dy = dy[:, :, None, :]

            dx = afb1d(dy, g0, g1, dim=3)

            dlow = dx[:, ::2, 0].contiguous()
            dhigh = dx[:, 1::2, 0].contiguous()
        return dlow, dhigh, None, None, None, None, None

class DWT1DForward(nn.Module):
    def __init__(self, J=1, wave='db1'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0, h1 = wave.dec_lo, wave.dec_hi
        else:
            assert len(wave) == 2
            h0, h1 = wave[0], wave[1]
        filts = prep_filt_afb1d(h0, h1)
        self.register_buffer('h0', filts[0])
        self.register_buffer('h1', filts[1])
        self.J = J

    def forward(self, x):
        assert x.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        highs = []
        x0 = x
        for j in range(self.J):
            x0, x1 = AFB1D.apply(x0, self.h0, self.h1)
            highs.append(x1)
        return x0, highs
        
class WPT1D(torch.nn.Module):
    """
    example usage:
    import torch
    from tft.wavelet import DWT1DForward, DWT1DInverse, WPT1D, IWPT1D
    x1d = torch.randn(2, 3, 4096)
    wt1d = DWT1DForward(wave='bior4.4')
    wpt1d = WPT1D(wt=wt1d, J=3)
    iwt1d = DWT1DInverse(wave='bior4.4')
    iwpt1d = IWPT1D(iwt=iwt1d, J=3)
    with torch.no_grad():
        X1d = wpt1d(x1d)
        xhat1d = iwpt1d(X1d)
    assert (xhat1d - x1d).abs().max() < 1e-5
    """
    def __init__(self, wt=DWT1DForward(wave='bior4.4'), J=4):
        super().__init__()
        self.wt = wt
        self.J = J

    def analysis_one_level(self,x):
        L, H = self.wt(x)
        X = torch.cat([L.unsqueeze(2),H[0].unsqueeze(2)],dim=2)
        X = einops.rearrange(X, 'b c f ℓ -> b (c f) ℓ')
        return X

    def wavelet_analysis(self, x, J):
        for _ in range(J):
            x = self.analysis_one_level(x)
        return x

    def forward(self, x):
        return self.wavelet_analysis(x, J=self.J)
        
class DWT1DInverse(nn.Module):
    def __init__(self, wave='db1'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, g1 = wave.rec_lo, wave.rec_hi
        else:
            assert len(wave) == 2
            g0, g1 = wave[0], wave[1]
        filts = prep_filt_sfb1d(g0, g1)
        self.register_buffer('g0', filts[0])
        self.register_buffer('g1', filts[1])

    def forward(self, coeffs):
        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            x0 = SFB1D.apply(x0, x1, self.g0, self.g1)
        return x0

class IWPT1D(torch.nn.Module):
    def __init__(self, iwt=DWT1DInverse(wave='bior4.4'), J=4):
        super().__init__()
        self.iwt = iwt
        self.J = J

    def synthesis_one_level(self, X):
        X = einops.rearrange(X, 'b (c f) ℓ -> b c f ℓ', f=2)
        L, H = torch.split(X, [1, 1], dim=2)
        L = L.squeeze(2)
        H = [H.squeeze(2)]
        y = self.iwt((L, H))
        return y

    def wavelet_synthesis(self, x, J):
        for _ in range(J):
            x = self.synthesis_one_level(x)
        return x

    def forward(self, x):
        return self.wavelet_synthesis(x, J=self.J)

def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
    h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
    if h0_row is None:
        h0_row, h1_row = h0_col, h1_col
    else:
        h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)

    h0_col = h0_col.reshape((1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, -1))
    return h0_col, h1_col, h0_row, h1_row

def afb2d(x, filts):
    tensorize = [not isinstance(f, torch.Tensor) for f in filts]
    if len(filts) == 2:
        h0, h1 = filts
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                h0, h1, device=x.device)
        else:
            h0_col = h0
            h0_row = h0.transpose(2,3)
            h1_col = h1
            h1_row = h1.transpose(2,3)
    elif len(filts) == 4:
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                *filts, device=x.device)
        else:
            h0_col, h1_col, h0_row, h1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lohi = afb1d(x, h0_row, h1_row, dim=3)
    y = afb1d(lohi, h0_col, h1_col, dim=2)

    return y

class AFB2D(Function):
    @staticmethod
    def forward(ctx, x, h0_row, h1_row, h0_col, h1_col):
        ctx.save_for_backward(h0_row, h1_row, h0_col, h1_col)
        ctx.shape = x.shape[-2:]
        lohi = afb1d(x, h0_row, h1_row, dim=3)
        y = afb1d(lohi, h0_col, h1_col, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        low = y[:,:,0].contiguous()
        highs = y[:,:,1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, low, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            lh, hl, hh = torch.unbind(highs, dim=2)
            lo = sfb1d(low, lh, h0_col, h1_col, dim=2)
            hi = sfb1d(hl, hh, h0_col, h1_col, dim=2)
            dx = sfb1d(lo, hi, h0_row, h1_row, dim=3)
            if dx.shape[-2] > ctx.shape[-2] and dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:ctx.shape[-2], :ctx.shape[-1]]
            elif dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:,:,:ctx.shape[-2]]
            elif dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:,:ctx.shape[-1]]
        return dx, None, None, None, None, None

def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
    g0_col, g1_col = prep_filt_sfb1d(g0_col, g1_col, device)
    if g0_row is None:
        g0_row, g1_row = g0_col, g1_col
    else:
        g0_row, g1_row = prep_filt_sfb1d(g0_row, g1_row, device)

    g0_col = g0_col.reshape((1, 1, -1, 1))
    g1_col = g1_col.reshape((1, 1, -1, 1))
    g0_row = g0_row.reshape((1, 1, 1, -1))
    g1_row = g1_row.reshape((1, 1, 1, -1))

    return g0_col, g1_col, g0_row, g1_row

def sfb2d(ll, lh, hl, hh, filts):
    tensorize = [not isinstance(x, torch.Tensor) for x in filts]
    if len(filts) == 2:
        g0, g1 = filts
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(g0, g1)
        else:
            g0_col = g0
            g0_row = g0.transpose(2,3)
            g1_col = g1
            g1_row = g1.transpose(2,3)
    elif len(filts) == 4:
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(*filts)
        else:
            g0_col, g1_col, g0_row, g1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lo = sfb1d(ll, lh, g0_col, g1_col, dim=2)
    hi = sfb1d(hl, hh, g0_col, g1_col, dim=2)
    y = sfb1d(lo, hi, g0_row, g1_row, dim=3)

    return y
        
class SFB2D(Function):
    @staticmethod
    def forward(ctx, low, highs, g0_row, g1_row, g0_col, g1_col):
        ctx.save_for_backward(g0_row, g1_row, g0_col, g1_col)

        lh, hl, hh = torch.unbind(highs, dim=2)
        lo = sfb1d(low, lh, g0_col, g1_col, dim=2)
        hi = sfb1d(hl, hh, g0_col, g1_col, dim=2)
        y = sfb1d(lo, hi, g0_row, g1_row, dim=3)
        return y

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            g0_row, g1_row, g0_col, g1_col = ctx.saved_tensors
            dx = afb1d(dy, g0_row, g1_row, dim=3)
            dx = afb1d(dx, g0_col, g1_col, dim=2)
            s = dx.shape
            dx = dx.reshape(s[0], -1, 4, s[-2], s[-1])
            dlow = dx[:,:,0].contiguous()
            dhigh = dx[:,:,1:].contiguous()
        return dlow, dhigh, None, None, None, None, None

class DWT2DForward(nn.Module):
    def __init__(self, J=1, wave='db1'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]
        filts = prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J

    def forward(self, x):
        yh = []
        ll = x
        for j in range(self.J):
            ll, high = AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row)
            yh.append(high)
        return ll, yh

class WPT2D(torch.nn.Module):
    """
    import torch
    from tft.wavelet import DWT2DForward, DWT2DInverse, WPT2D, IWPT2D
    x2d = torch.randn(2, 3, 64, 64)
    wt2d = DWT2DForward(wave='bior4.4')
    wpt2d = WPT2D(wt=wt2d, J=3)
    iwt2d = DWT2DInverse(wave='bior4.4')
    iwpt2d = IWPT2D(iwt=iwt2d, J=3)
    with torch.no_grad():
        X2d = wpt2d(x2d)
        xhat2d = iwpt2d(X2d)
    assert (xhat2d - x2d).abs().max() < 1e-5
    """
    def __init__(self, wt=DWT2DForward(wave='bior4.4'), J=4):
        super().__init__()
        self.wt  = wt
        self.J = J
    def analysis_one_level(self,x):
        L, H = self.wt(x)
        X = torch.cat([L.unsqueeze(2),H[0]],dim=2)
        X = einops.rearrange(X, 'b c f h w -> b (c f) h w')
        return X
    def wavelet_analysis(self,x,J):
        for _ in range(J):
            x = self.analysis_one_level(x)
        return x
    def forward(self, x):
        return self.wavelet_analysis(x,J=self.J)

class DWT2DInverse(nn.Module):
    def __init__(self, wave='db1'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        filts = prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_col', filts[0])
        self.register_buffer('g1_col', filts[1])
        self.register_buffer('g0_row', filts[2])
        self.register_buffer('g1_row', filts[3])

    def forward(self, coeffs):
        yl, yh = coeffs
        ll = yl
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2], ll.shape[-1], device=ll.device)
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            ll = SFB2D.apply(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row)
        return ll

class IWPT2D(torch.nn.Module):
    def __init__(self, iwt=DWT2DInverse(wave='bior4.4'), J=4):
        super().__init__()
        self.iwt  = iwt
        self.J = J
    def synthesis_one_level(self,X):
        X = einops.rearrange(X, 'b (c f) h w -> b c f h w', f=4)
        L, H = torch.split(X, [1, 3], dim=2)
        L = L.squeeze(2)
        H = [H]
        y = self.iwt((L, H))
        return y
    def wavelet_synthesis(self,x,J):
        for _ in range(J):
            x = self.synthesis_one_level(x)
        return x
    def forward(self, x):
        return self.wavelet_synthesis(x,J=self.J)

def prep_filt_afb3d(h0_d, h1_d, h0_h=None, h1_h=None, h0_w=None, h1_w=None, device=None):
    # Prepare depth filters
    h0_d, h1_d = prep_filt_afb1d(h0_d, h1_d, device)
    # If height filters are not provided, use depth filters
    if h0_h is None:
        h0_h, h1_h = h0_d, h1_d
    else:
        h0_h, h1_h = prep_filt_afb1d(h0_h, h1_h, device)
    # If width filters are not provided, use height filters
    if h0_w is None:
        h0_w, h1_w = h0_h, h1_h
    else:
        h0_w, h1_w = prep_filt_afb1d(h0_w, h1_w, device)
    
    # Reshape filters for 3D convolution: (out_channels, in_channels/groups, D, H, W)
    h0_d = h0_d.reshape(1, 1, -1, 1, 1)  # Depth filter
    h1_d = h1_d.reshape(1, 1, -1, 1, 1)
    h0_h = h0_h.reshape(1, 1, 1, -1, 1)  # Height filter
    h1_h = h1_h.reshape(1, 1, 1, -1, 1)
    h0_w = h0_w.reshape(1, 1, 1, 1, -1)  # Width filter
    h1_w = h1_w.reshape(1, 1, 1, 1, -1)
    return h0_d, h1_d, h0_h, h1_h, h0_w, h1_w

def afb1d_3d(x, h0, h1, dim=-1):
    assert x.ndim == 5, "Input must be 5D (N, C, D, H, W)"
    assert dim in [2, 3, 4], "dim must be 2 (D), 3 (H), or 4 (W)"
    
    C = x.shape[1]
    N = x.shape[dim]
    L = h0.numel()
    L2 = L // 2
    
    # Prepare filter shapes based on dimension
    if dim == 2:  # Depth
        h0 = h0.view(1, 1, L, 1, 1)
        h1 = h1.view(1, 1, L, 1, 1)
        padding = (L-1, 0, 0)  # padD, padH, padW
        stride = (2, 1, 1)
    elif dim == 3:  # Height
        h0 = h0.view(1, 1, 1, L, 1)
        h1 = h1.view(1, 1, 1, L, 1)
        padding = (0, L-1, 0)
        stride = (1, 2, 1)
    elif dim == 4:  # Width
        h0 = h0.view(1, 1, 1, 1, L)
        h1 = h1.view(1, 1, 1, 1, L)
        padding = (0, 0, L-1)
        stride = (1, 1, 2)
    
    # Stack filters for each channel
    h = torch.cat([h0, h1] * C, dim=0)  # Shape: (2*C, 1, kernel_D, kernel_H, kernel_W)
    
    # Pad to handle odd lengths
    if N % 2 == 1:
        if dim == 2:
            x = torch.cat((x, x[:, :, -1:, :, :]), dim=2)
        elif dim == 3:
            x = torch.cat((x, x[:, :, :, -1:, :]), dim=3)
        elif dim == 4:
            x = torch.cat((x, x[:, :, :, :, -1:]), dim=4)
        N += 1
    
    # Roll the input to align with filter
    x = torch.roll(x, -L2, dims=dim)
    
    # Apply 3D convolution with asymmetric padding
    lohi = F.conv3d(x, h, stride=stride, padding=padding, groups=C)
    
    # Handle overlap if output is larger than N//2
    N2 = N // 2
    if lohi.shape[dim] > N2:
        if dim == 2:
            lohi[:, :, :L2, :, :] += lohi[:, :, N2:N2+L2, :, :]
            lohi = lohi[:, :, :N2, :, :]
        elif dim == 3:
            lohi[:, :, :, :L2, :] += lohi[:, :, :, N2:N2+L2, :]
            lohi = lohi[:, :, :, :N2, :]
        elif dim == 4:
            lohi[:, :, :, :, :L2] += lohi[:, :, :, :, N2:N2+L2]
            lohi = lohi[:, :, :, :, :N2]
    
    return lohi

def afb3d(x, filts):
    tensorize = [not isinstance(f, torch.Tensor) for f in filts]
    if len(filts) == 2:
        h0, h1 = filts
        if True in tensorize:
            h0_d, h1_d, h0_h, h1_h, h0_w, h1_w = prep_filt_afb3d(h0, h1, device=x.device)
        else:
            h0_d = h0.transpose(2, 4)
            h1_d = h1.transpose(2, 4)
            h0_h = h0.transpose(2, 3)
            h1_h = h1.transpose(2, 3)
            h0_w = h0
            h1_w = h1
    elif len(filts) == 6:
        if True in tensorize:
            h0_d, h1_d, h0_h, h1_h, h0_w, h1_w = prep_filt_afb3d(*filts, device=x.device)
        else:
            h0_d, h1_d, h0_h, h1_h, h0_w, h1_w = filts
    else:
        raise ValueError("filts must be length 2 or 6")
    
    # Apply decomposition sequentially
    lohi_w = afb1d_3d(x, h0_w, h1_w, dim=4)      # (N, 2*C, D, H, W/2)
    lohi_hw = afb1d_3d(lohi_w, h0_h, h1_h, dim=3) # (N, 4*C, D, H/2, W/2)
    y = afb1d_3d(lohi_hw, h0_d, h1_d, dim=2)     # (N, 8*C, D/2, H/2, W/2)
    
    return y

class DWT3DForward(nn.Module):
    def __init__(self, J=1, wave='db1'):
        super().__init__()
        self.J = J
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0, h1 = wave.dec_lo, wave.dec_hi
            h0_d, h1_d = h0, h1
            h0_h, h1_h = h0, h1
            h0_w, h1_w = h0, h1
        else:
            if len(wave) == 2:
                h0, h1 = wave[0], wave[1]
                h0_d, h1_d = h0, h1
                h0_h, h1_h = h0, h1
                h0_w, h1_w = h0, h1
            elif len(wave) == 6:
                h0_d, h1_d, h0_h, h1_h, h0_w, h1_w = wave
            else:
                raise ValueError("wave must be a string, pywt.Wavelet, or tuple of 2 or 6 filters")
        
        filts = prep_filt_afb3d(h0_d, h1_d, h0_h, h1_h, h0_w, h1_w)
        self.register_buffer('h0_d', filts[0])
        self.register_buffer('h1_d', filts[1])
        self.register_buffer('h0_h', filts[2])
        self.register_buffer('h1_h', filts[3])
        self.register_buffer('h0_w', filts[4])
        self.register_buffer('h1_w', filts[5])

    def forward(self, x):
        assert x.ndim == 5, "Input must be 5D (N, C, D, H, W)"
        yhs = []
        ll = x
        filts = (self.h0_d, self.h1_d, self.h0_h, self.h1_h, self.h0_w, self.h1_w)
        for j in range(self.J):
            y = afb3d(ll, filts)  # (N, 8*C, D/2^j, H/2^j, W/2^j)
            C = ll.shape[1]
            ll = y[:, :C]         # LLL subband
            high = y[:, C:]       # 7 high-pass subbands
            yhs.append(high)
        return ll, yhs


class WPT3D(nn.Module):
    """
    import torch
    from tft.wavelet import DWT3DForward, DWT3DInverse, WPT3D, IWPT3D
    x3d = torch.randn(2, 3, 16, 16, 16)
    wt3d = DWT3DForward(wave='bior4.4')
    wpt3d = WPT3D(wt=wt3d,J=1)
    iwt3d = DWT3DInverse(wave='bior4.4')
    iwpt3d = IWPT3D(iwt=iwt3d, J=1)
    with torch.no_grad():
        X3d = wpt3d(x3d)
        xhat3d = iwpt3d(X3d)
    assert (xhat3d - x3d).abs().max() < 1e-5
    """
    def __init__(self, wt=None, J=4, wave='db1'):
        super().__init__()
        self.J = J
        if wt is None:
            self.wt = DWT3DForward(J=1, wave=wave)
        else:
            self.wt = wt

    def analysis_one_level(self, x):
        L, H = self.wt(x)
        # Concatenate low and high subbands: 8 subbands total
        X = torch.cat([L.unsqueeze(2), H[0].view(L.shape[0], -1, 7, L.shape[2], L.shape[3], L.shape[4])], dim=2)
        X = einops.rearrange(X, 'b c f d h w -> b (c f) d h w')
        return X

    def forward(self, x):
        for _ in range(self.J):
            x = self.analysis_one_level(x)
        return x

def sfb1d_3d(lo, hi, g0, g1, dim=-1):
    assert lo.ndim == 5 and hi.ndim == 5, "Inputs must be 5D (N, C, D, H, W)"
    assert lo.shape == hi.shape, "Low and high subbands must have the same shape"
    assert dim in [2, 3, 4], "dim must be 2 (D), 3 (H), or 4 (W)"

    C = lo.shape[1]
    d = dim % 5
    N = 2 * lo.shape[d]  # Output size after upsampling
    L = g0.numel() if isinstance(g0, torch.Tensor) else len(g0)
    L2 = L // 2

    # Convert filters to tensors if necessary
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.array(g0).ravel(), dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.array(g1).ravel(), dtype=torch.float, device=hi.device)

    # Reshape filters based on dimension
    if d == 2:  # Depth
        g0 = g0.reshape(1, 1, -1, 1, 1)
        g1 = g1.reshape(1, 1, -1, 1, 1)
        stride = (2, 1, 1)
    elif d == 3:  # Height
        g0 = g0.reshape(1, 1, 1, -1, 1)
        g1 = g1.reshape(1, 1, 1, -1, 1)
        stride = (1, 2, 1)
    elif d == 4:  # Width
        g0 = g0.reshape(1, 1, 1, 1, -1)
        g1 = g1.reshape(1, 1, 1, 1, -1)
        stride = (1, 1, 2)

    # Repeat filters for each channel
    g0 = g0.repeat(C, 1, 1, 1, 1)
    g1 = g1.repeat(C, 1, 1, 1, 1)

    # Apply transposed convolution (upsampling)
    y = F.conv_transpose3d(lo, g0, stride=stride, groups=C) + \
        F.conv_transpose3d(hi, g1, stride=stride, groups=C)

    # Handle boundary overlap
    if y.shape[d] > N:
        idx = [slice(None)] * 5
        idx[d] = slice(None, L-2)
        idx2 = [slice(None)] * 5
        idx2[d] = slice(N, N + L-2)
        y = y.clone()  # Avoid in-place modification
        y[tuple(idx)] = y[tuple(idx)] + y[tuple(idx2)]
        idx[d] = slice(None, N)
        y = y[tuple(idx)]

    # Phase adjustment
    y = roll(y, 1 - L2, dim=d)

    return y

def sfb3d(y, filts):
    assert y.ndim == 5 and y.shape[1] % 8 == 0, "Input must be (N, 8*C, D/2, H/2, W/2)"
    C = y.shape[1] // 8
    g0_d, g1_d, g0_h, g1_h, g0_w, g1_w = filts

    # Extract the eight subbands for each original channel
    y0 = y[:, 0::8, :, :, :]  # LLL: lo_d lo_h lo_w
    y1 = y[:, 1::8, :, :, :]  # HLL: hi_d lo_h lo_w
    y2 = y[:, 2::8, :, :, :]  # LHL: lo_d hi_h lo_w
    y3 = y[:, 3::8, :, :, :]  # HHL: hi_d hi_h lo_w
    y4 = y[:, 4::8, :, :, :]  # LLH: lo_d lo_h hi_w
    y5 = y[:, 5::8, :, :, :]  # HLH: hi_d lo_h hi_w
    y6 = y[:, 6::8, :, :, :]  # LHH: lo_d hi_h hi_w
    y7 = y[:, 7::8, :, :, :]  # HHH: hi_d hi_h hi_w

    # Step 1: Reconstruct along depth
    t0 = sfb1d_3d(y0, y1, g0_d, g1_d, dim=2)  # lo_h lo_w (N, C, D, H/2, W/2)
    t1 = sfb1d_3d(y2, y3, g0_d, g1_d, dim=2)  # hi_h lo_w
    t2 = sfb1d_3d(y4, y5, g0_d, g1_d, dim=2)  # lo_h hi_w
    t3 = sfb1d_3d(y6, y7, g0_d, g1_d, dim=2)  # hi_h hi_w

    # Step 2: Reconstruct along height
    u0 = sfb1d_3d(t0, t1, g0_h, g1_h, dim=3)  # lo_w (N, C, D, H, W/2)
    u1 = sfb1d_3d(t2, t3, g0_h, g1_h, dim=3)  # hi_w

    # Step 3: Reconstruct along width
    y_rec = sfb1d_3d(u0, u1, g0_w, g1_w, dim=4)  # (N, C, D, H, W)

    return y_rec

class DWT3DInverse(nn.Module):
    def __init__(self, wave='db1'):
        super().__init__()
        # Filter initialization
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, g1 = wave.rec_lo, wave.rec_hi
            g0_d, g1_d = g0, g1
            g0_h, g1_h = g0, g1
            g0_w, g1_w = g0, g1
        else:
            if len(wave) == 2:
                g0, g1 = wave[0], wave[1]
                g0_d, g1_d = g0, g1
                g0_h, g1_h = g0, g1
                g0_w, g1_w = g0, g1
            elif len(wave) == 6:
                g0_d, g1_d, g0_h, g1_h, g0_w, g1_w = wave
            else:
                raise ValueError("wave must be a string, pywt.Wavelet, or tuple of 2 or 6 filters")

        # Register filters as buffers
        self.register_buffer('g0_d', torch.tensor(g0_d, dtype=torch.float).reshape(1, 1, -1, 1, 1))
        self.register_buffer('g1_d', torch.tensor(g1_d, dtype=torch.float).reshape(1, 1, -1, 1, 1))
        self.register_buffer('g0_h', torch.tensor(g0_h, dtype=torch.float).reshape(1, 1, 1, -1, 1))
        self.register_buffer('g1_h', torch.tensor(g1_h, dtype=torch.float).reshape(1, 1, 1, -1, 1))
        self.register_buffer('g0_w', torch.tensor(g0_w, dtype=torch.float).reshape(1, 1, 1, 1, -1))
        self.register_buffer('g1_w', torch.tensor(g1_w, dtype=torch.float).reshape(1, 1, 1, 1, -1))

    def forward(self, coeffs):
        yl, yh = coeffs
        ll = yl

        filts = (self.g0_d, self.g1_d, self.g0_h, self.g1_h, self.g0_w, self.g1_w)
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], 7 * ll.shape[1], ll.shape[2],
                                ll.shape[3], ll.shape[4], device=ll.device)
            # Ensure spatial dimensions match by trimming if necessary
            dims = [-3, -2, -1]  # D, H, W
            for d in dims:
                if ll.shape[d] > h.shape[d]:
                    ll = ll.narrow(d, 0, h.shape[d])
            y_all = torch.cat([ll, h], dim=1)  # (N, 8*C, D/2, H/2, W/2)
            ll = sfb3d(y_all, filts)

        return ll

class IWPT3D(nn.Module):
    def __init__(self, iwt=None, J=4, wave='db1'):
        super().__init__()
        self.J = J
        if iwt is None:
            self.iwt = DWT3DInverse(wave=wave)
        else:
            self.iwt = iwt

    def synthesis_one_level(self, X):
        X = einops.rearrange(X, 'b (c f) d h w -> b c f d h w', f=8)
        L, H = torch.split(X, [1, 7], dim=2)  # L: (N, C, 1, D/2, H/2, W/2), H: (N, C, 7, D/2, H/2, W/2)
        L = L.squeeze(2)  # (N, C, D/2, H/2, W/2)
        H = einops.rearrange(H, 'b c f d h w -> b (c f) d h w')  # (N, 7*C, D/2, H/2, W/2)
        y = self.iwt((L, [H]))
        return y

    def wavelet_synthesis(self, x, J):
        for _ in range(J):
            x = self.synthesis_one_level(x)
        return x

    def forward(self, x):
        return self.wavelet_synthesis(x, self.J)