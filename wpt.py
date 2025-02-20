# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import einops
from torch.autograd import Function


# In[2]:

#TODO: this might need to be updated to support 3d
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
    elif dim == 2 or dim == -2:
        return torch.cat((x[:,:,-n:], x[:,:,:-n+end]), dim=2)
    elif dim == 3 or dim == -1:
        return torch.cat((x[:,:,:,-n:], x[:,:,:,:-n+end]), dim=3)


# In[3]:


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


# In[4]:


x1d = torch.randn(2, 3, 4096)
wt1d = DWT1DForward(wave='bior4.4')
wpt1d = WPT1D(wt=wt1d, J=3)
iwt1d = DWT1DInverse(wave='bior4.4')
iwpt1d = IWPT1D(iwt=iwt1d, J=3)
with torch.no_grad():
    X1d = wpt1d(x1d)
    xhat1d = iwpt1d(X1d)
assert (xhat1d - x1d).abs().max() < 1e-5


# In[5]:


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


# In[6]:


x2d = torch.randn(2, 3, 64, 64)
wt2d = DWT2DForward(wave='bior4.4')
wpt2d = WPT2D(wt=wt2d, J=3)
iwt2d = DWT2DInverse(wave='bior4.4')
iwpt2d = IWPT2D(iwt=iwt2d, J=3)
with torch.no_grad():
    X2d = wpt2d(x2d)
    xhat2d = iwpt2d(X2d)
assert (xhat2d - x2d).abs().max() < 1e-5


# In[7]:


# TODO: use the 2d version as a starting point and impelement a 3d version

# def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
#     h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
#     if h0_row is None:
#         h0_row, h1_row = h0_col, h1_col
#     else:
#         h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)
#     h0_col = h0_col.reshape((1, 1, -1, 1))
#     h1_col = h1_col.reshape((1, 1, -1, 1))
#     h0_row = h0_row.reshape((1, 1, 1, -1))
#     h1_row = h1_row.reshape((1, 1, 1, -1))
#     return h0_col, h1_col, h0_row, h1_row

def prep_filt_afb3d():

# TODO: use the 2d version as a starting point and impelement a 3d version

# def afb2d(x, filts):
#     tensorize = [not isinstance(f, torch.Tensor) for f in filts]
#     if len(filts) == 2:
#         h0, h1 = filts
#         if True in tensorize:
#             h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
#                 h0, h1, device=x.device)
#         else:
#             h0_col = h0
#             h0_row = h0.transpose(2,3)
#             h1_col = h1
#             h1_row = h1.transpose(2,3)
#     elif len(filts) == 4:
#         if True in tensorize:
#             h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
#                 *filts, device=x.device)
#         else:
#             h0_col, h1_col, h0_row, h1_row = filts
#     else:
#         raise ValueError("Unknown form for input filts")
#     lohi = afb1d(x, h0_row, h1_row, dim=3)
#     y = afb1d(lohi, h0_col, h1_col, dim=2)
#     return y

def afb3d():

# TODO: use the 2d version as a starting point and impelement a 3d version

# class AFB2D(Function):
#     @staticmethod
#     def forward(ctx, x, h0_row, h1_row, h0_col, h1_col):
#         ctx.save_for_backward(h0_row, h1_row, h0_col, h1_col)
#         ctx.shape = x.shape[-2:]
#         lohi = afb1d(x, h0_row, h1_row, dim=3)
#         y = afb1d(lohi, h0_col, h1_col, dim=2)
#         s = y.shape
#         y = y.reshape(s[0], -1, 4, s[-2], s[-1])
#         low = y[:,:,0].contiguous()
#         highs = y[:,:,1:].contiguous()
#         return low, highs
#     @staticmethod
#     def backward(ctx, low, highs):
#         dx = None
#         if ctx.needs_input_grad[0]:
#             h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
#             lh, hl, hh = torch.unbind(highs, dim=2)
#             lo = sfb1d(low, lh, h0_col, h1_col, dim=2)
#             hi = sfb1d(hl, hh, h0_col, h1_col, dim=2)
#             dx = sfb1d(lo, hi, h0_row, h1_row, dim=3)
#             if dx.shape[-2] > ctx.shape[-2] and dx.shape[-1] > ctx.shape[-1]:
#                 dx = dx[:,:,:ctx.shape[-2], :ctx.shape[-1]]
#             elif dx.shape[-2] > ctx.shape[-2]:
#                 dx = dx[:,:,:ctx.shape[-2]]
#             elif dx.shape[-1] > ctx.shape[-1]:
#                 dx = dx[:,:,:,:ctx.shape[-1]]
#         return dx, None, None, None, None, None

class AFB3D(Function):

# TODO: use the 2d version as a starting point and impelement a 3d version

# def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
#     g0_col, g1_col = prep_filt_sfb1d(g0_col, g1_col, device)
#     if g0_row is None:
#         g0_row, g1_row = g0_col, g1_col
#     else:
#         g0_row, g1_row = prep_filt_sfb1d(g0_row, g1_row, device)
#     g0_col = g0_col.reshape((1, 1, -1, 1))
#     g1_col = g1_col.reshape((1, 1, -1, 1))
#     g0_row = g0_row.reshape((1, 1, 1, -1))
#     g1_row = g1_row.reshape((1, 1, 1, -1))
#     return g0_col, g1_col, g0_row, g1_row

def prep_filt_sfb3d():

# TODO: use the 2d version as a starting point and impelement a 3d version

# def sfb2d(ll, lh, hl, hh, filts):
#     tensorize = [not isinstance(x, torch.Tensor) for x in filts]
#     if len(filts) == 2:
#         g0, g1 = filts
#         if True in tensorize:
#             g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(g0, g1)
#         else:
#             g0_col = g0
#             g0_row = g0.transpose(2,3)
#             g1_col = g1
#             g1_row = g1.transpose(2,3)
#     elif len(filts) == 4:
#         if True in tensorize:
#             g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(*filts)
#         else:
#             g0_col, g1_col, g0_row, g1_row = filts
#     else:
#         raise ValueError("Unknown form for input filts")
#     lo = sfb1d(ll, lh, g0_col, g1_col, dim=2)
#     hi = sfb1d(hl, hh, g0_col, g1_col, dim=2)
#     y = sfb1d(lo, hi, g0_row, g1_row, dim=3)
#     return y

def sfb3d():

# TODO: use the 2d version as a starting point and impelement a 3d version

# class SFB2D(Function):
#     @staticmethod
#     def forward(ctx, low, highs, g0_row, g1_row, g0_col, g1_col):
#         ctx.save_for_backward(g0_row, g1_row, g0_col, g1_col)
#         lh, hl, hh = torch.unbind(highs, dim=2)
#         lo = sfb1d(low, lh, g0_col, g1_col, dim=2)
#         hi = sfb1d(hl, hh, g0_col, g1_col, dim=2)
#         y = sfb1d(lo, hi, g0_row, g1_row, dim=3)
#         return y
#     @staticmethod
#     def backward(ctx, dy):
#         dlow, dhigh = None, None
#         if ctx.needs_input_grad[0]:
#             g0_row, g1_row, g0_col, g1_col = ctx.saved_tensors
#             dx = afb1d(dy, g0_row, g1_row, dim=3)
#             dx = afb1d(dx, g0_col, g1_col, dim=2)
#             s = dx.shape
#             dx = dx.reshape(s[0], -1, 4, s[-2], s[-1])
#             dlow = dx[:,:,0].contiguous()
#             dhigh = dx[:,:,1:].contiguous()
#         return dlow, dhigh, None, None, None, None, None

class SFB3D(Function):

# TODO: use the 2d version as a starting point and impelement a 3d version
    
# class DWT2DForward(nn.Module):
#     def __init__(self, J=1, wave='db1'):
#         super().__init__()
#         if isinstance(wave, str):
#             wave = pywt.Wavelet(wave)
#         if isinstance(wave, pywt.Wavelet):
#             h0_col, h1_col = wave.dec_lo, wave.dec_hi
#             h0_row, h1_row = h0_col, h1_col
#         else:
#             if len(wave) == 2:
#                 h0_col, h1_col = wave[0], wave[1]
#                 h0_row, h1_row = h0_col, h1_col
#             elif len(wave) == 4:
#                 h0_col, h1_col = wave[0], wave[1]
#                 h0_row, h1_row = wave[2], wave[3]
#         filts = prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
#         self.register_buffer('h0_col', filts[0])
#         self.register_buffer('h1_col', filts[1])
#         self.register_buffer('h0_row', filts[2])
#         self.register_buffer('h1_row', filts[3])
#         self.J = J
#     def forward(self, x):
#         yh = []
#         ll = x
#         for j in range(self.J):
#             ll, high = AFB2D.apply(
#                 ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row)
#             yh.append(high)
#         return ll, yh

class DWT3DForward(nn.Module):

# TODO: use the 2d version as a starting point and impelement a 3d version
    
# class WPT2D(torch.nn.Module):
#     def __init__(self, wt=DWT2DForward(wave='bior4.4'), J=4):
#         super().__init__()
#         self.wt  = wt
#         self.J = J
#     def analysis_one_level(self,x):
#         L, H = self.wt(x)
#         X = torch.cat([L.unsqueeze(2),H[0]],dim=2)
#         X = einops.rearrange(X, 'b c f h w -> b (c f) h w')
#         return X
#     def wavelet_analysis(self,x,J):
#         for _ in range(J):
#             x = self.analysis_one_level(x)
#         return x
#     def forward(self, x):
#         return self.wavelet_analysis(x,J=self.J)

class WPT3D(torch.nn.Module):

# TODO: use the 2d version as a starting point and impelement a 3d version
    
# class DWT2DInverse(nn.Module):
#     def __init__(self, wave='db1'):
#         super().__init__()
#         if isinstance(wave, str):
#             wave = pywt.Wavelet(wave)
#         if isinstance(wave, pywt.Wavelet):
#             g0_col, g1_col = wave.rec_lo, wave.rec_hi
#             g0_row, g1_row = g0_col, g1_col
#         else:
#             if len(wave) == 2:
#                 g0_col, g1_col = wave[0], wave[1]
#                 g0_row, g1_row = g0_col, g1_col
#             elif len(wave) == 4:
#                 g0_col, g1_col = wave[0], wave[1]
#                 g0_row, g1_row = wave[2], wave[3]
#         filts = prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
#         self.register_buffer('g0_col', filts[0])
#         self.register_buffer('g1_col', filts[1])
#         self.register_buffer('g0_row', filts[2])
#         self.register_buffer('g1_row', filts[3])
#     def forward(self, coeffs):
#         yl, yh = coeffs
#         ll = yl
#         for h in yh[::-1]:
#             if h is None:
#                 h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2], ll.shape[-1], device=ll.device)
#             if ll.shape[-2] > h.shape[-2]:
#                 ll = ll[...,:-1,:]
#             if ll.shape[-1] > h.shape[-1]:
#                 ll = ll[...,:-1]
#             ll = SFB2D.apply(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row)
#         return ll

DWT3DInverse(nn.Module):

# TODO: use the 2d version as a starting point and impelement a 3d version
    
# class IWPT2D(torch.nn.Module):
#     def __init__(self, iwt=DWT2DInverse(wave='bior4.4'), J=4):
#         super().__init__()
#         self.iwt  = iwt
#         self.J = J
#     def synthesis_one_level(self,X):
#         X = einops.rearrange(X, 'b (c f) h w -> b c f h w', f=4)
#         L, H = torch.split(X, [1, 3], dim=2)
#         L = L.squeeze(2)
#         H = [H]
#         y = self.iwt((L, H))
#         return y
#     def wavelet_synthesis(self,x,J):
#         for _ in range(J):
#             x = self.synthesis_one_level(x)
#         return x
#     def forward(self, x):
#         return self.wavelet_synthesis(x,J=self.J)

IWPT3D(torch.nn.Module):

# In[8]:


x3d = torch.randn(2, 3, 16, 16, 16)
wt3d = DWT3DForward(wave='bior4.4')
wpt3d = WPT3D(wt=wt3d, J=3)
iwt3d = DWT3DInverse(wave='bior4.4')
iwpt3d = IWPT3D(iwt=iwt3d, J=3)
with torch.no_grad():
    X3d = wpt3d(x3d)
    xhat3d = iwpt3d(X3d)
assert (xhat3d - x3d).abs().max() < 1e-5

