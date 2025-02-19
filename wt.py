# Adapted from https://github.com/fbcotter/pytorch_wavelets

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import pywt

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

def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

def prep_filt_afb1d(h0, h1, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.

    Inputs:
        h0 (array-like): low pass column filter bank
        h1 (array-like): high pass column filter bank
        device: which device to put the tensors on to

    Returns:
        (h0, h1)
    """
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
    return h0, h1

def afb1d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image

    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).

    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
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

    if mode == 'per' or mode == 'periodization':
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
    else:
        # Calculate the pad size
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = F.pad(x, pad)
            pad = (p//2, 0) if d == 2 else (0, p//2)
            # Calculate the high and lowpass
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (0, 0, p//2, (p+1)//2) if d == 2 else (p//2, (p+1)//2, 0, 0)
            x = mypad(x, pad=pad, mode=mode)
            lohi = F.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return lohi
        
class AFB1D(Function):
    """ Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0: lowpass
        h1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        x0: Tensor of shape (N, C, L') - lowpass
        x1: Tensor of shape (N, C, L') - highpass
    """
    @staticmethod
    def forward(ctx, x, h0, h1, mode):
        mode = int_to_mode(mode)

        # Make inputs 4d
        x = x[:, :, None, :]
        h0 = h0[:, :, None, :]
        h1 = h1[:, :, None, :]

        # Save for backwards
        ctx.save_for_backward(h0, h1)
        ctx.shape = x.shape[3]
        ctx.mode = mode

        lohi = afb1d(x, h0, h1, mode=mode, dim=3)
        x0 = lohi[:, ::2, 0].contiguous()
        x1 = lohi[:, 1::2, 0].contiguous()
        return x0, x1

    @staticmethod
    def backward(ctx, dx0, dx1):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0, h1 = ctx.saved_tensors

            # Make grads 4d
            dx0 = dx0[:, :, None, :]
            dx1 = dx1[:, :, None, :]

            dx = sfb1d(dx0, dx1, h0, h1, mode=mode, dim=3)[:, :, 0]

            # Check for odd input
            if dx.shape[2] > ctx.shape:
                dx = dx[:, :, :ctx.shape]

        return dx, None, None, None, None, None

def prep_filt_sfb1d(g0, g1, device=None):
    """
    Prepares the filters to be of the right form for the sfb1d function. In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0 (array-like): low pass filter bank
        g1 (array-like): high pass filter bank
        device: which device to put the tensors on to

    Returns:
        (g0, g1)
    """
    g0 = np.array(g0).ravel()
    g1 = np.array(g1).ravel()
    t = torch.get_default_dtype()
    g0 = torch.tensor(g0, device=device, dtype=t).reshape((1, 1, -1))
    g1 = torch.tensor(g1, device=device, dtype=t).reshape((1, 1, -1))

    return g0, g1

def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 4
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
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
    if mode == 'per' or mode == 'periodization':
        y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
            F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:,:,:L-2] = y[:,:,:L-2] + y[:,:,N:N+L-2]
            y = y[:,:,:N]
        else:
            y[:,:,:,:L-2] = y[:,:,:,:L-2] + y[:,:,:,N:N+L-2]
            y = y[:,:,:,:N]
        y = roll(y, 1-L//2, dim=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or \
                mode == 'periodic':
            pad = (L-2, 0) if d == 2 else (0, L-2)
            y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return y

class SFB1D(Function):
    """ Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        low (torch.Tensor): Lowpass to reconstruct of shape (N, C, L)
        high (torch.Tensor): Highpass to reconstruct of shape (N, C, L)
        g0: lowpass
        g1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*2, L')
    """
    @staticmethod
    def forward(ctx, low, high, g0, g1, mode):
        mode = int_to_mode(mode)
        # Make into a 2d tensor with 1 row
        low = low[:, :, None, :]
        high = high[:, :, None, :]
        g0 = g0[:, :, None, :]
        g1 = g1[:, :, None, :]

        ctx.mode = mode
        ctx.save_for_backward(g0, g1)

        return sfb1d(low, high, g0, g1, mode=mode, dim=3)[:, :, 0]

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            g0, g1, = ctx.saved_tensors
            dy = dy[:, :, None, :]

            dx = afb1d(dy, g0, g1, mode=mode, dim=3)

            dlow = dx[:, ::2, 0].contiguous()
            dhigh = dx[:, 1::2, 0].contiguous()
        return dlow, dhigh, None, None, None, None, None

class DWT1DForward(nn.Module):
    def __init__(self, J=1, wave='db1', mode='zero'):
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
        self.mode = mode

    def forward(self, x):
        assert x.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        highs = []
        x0 = x
        m_int = mode_to_int(self.mode)
        for j in range(self.J):
            x0, x1 = AFB1D.apply(x0, self.h0, self.h1, m_int)
            highs.append(x1)
        return x0, highs

class DWT1DInverse(nn.Module):
    def __init__(self, wave='db1', mode='zero'):
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
        self.mode = mode

    def forward(self, coeffs):
        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        m_int = mode_to_int(self.mode)
        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            x0 = SFB1D.apply(x0, x1, self.g0, self.g1, m_int)
        return x0

def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.

    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (h0_col, h1_col, h0_row, h1_row)
    """
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

def afb2d(x, filts, mode='zero'):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.

    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
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

    lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
    y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)

    return y

class AFB2D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
    @staticmethod
    def forward(ctx, x, h0_row, h1_row, h0_col, h1_col, mode):
        ctx.save_for_backward(h0_row, h1_row, h0_col, h1_col)
        ctx.shape = x.shape[-2:]
        mode = int_to_mode(mode)
        ctx.mode = mode
        lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
        y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        low = y[:,:,0].contiguous()
        highs = y[:,:,1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, low, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            lh, hl, hh = torch.unbind(highs, dim=2)
            lo = sfb1d(low, lh, h0_col, h1_col, mode=mode, dim=2)
            hi = sfb1d(hl, hh, h0_col, h1_col, mode=mode, dim=2)
            dx = sfb1d(lo, hi, h0_row, h1_row, mode=mode, dim=3)
            if dx.shape[-2] > ctx.shape[-2] and dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:ctx.shape[-2], :ctx.shape[-1]]
            elif dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:,:,:ctx.shape[-2]]
            elif dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:,:ctx.shape[-1]]
        return dx, None, None, None, None, None

def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the sfb2d function.  In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (g0_col, g1_col, g0_row, g1_row)
    """
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

def sfb2d(ll, lh, hl, hh, filts, mode='zero'):
    """ Does a single level 2d wavelet reconstruction of wavelet coefficients.
    Does separate row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.sfb1d`

    Inputs:
        ll (torch.Tensor): lowpass coefficients
        lh (torch.Tensor): horizontal coefficients
        hl (torch.Tensor): vertical coefficients
        hh (torch.Tensor): diagonal coefficients
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
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

    lo = sfb1d(ll, lh, g0_col, g1_col, mode=mode, dim=2)
    hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
    y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)

    return y
        
class SFB2D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
    @staticmethod
    def forward(ctx, low, highs, g0_row, g1_row, g0_col, g1_col, mode):
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(g0_row, g1_row, g0_col, g1_col)

        lh, hl, hh = torch.unbind(highs, dim=2)
        lo = sfb1d(low, lh, g0_col, g1_col, mode=mode, dim=2)
        hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
        y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)
        return y

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            g0_row, g1_row, g0_col, g1_col = ctx.saved_tensors
            dx = afb1d(dy, g0_row, g1_row, mode=mode, dim=3)
            dx = afb1d(dx, g0_col, g1_col, mode=mode, dim=2)
            s = dx.shape
            dx = dx.reshape(s[0], -1, 4, s[-2], s[-1])
            dlow = dx[:,:,0].contiguous()
            dhigh = dx[:,:,1:].contiguous()
        return dlow, dhigh, None, None, None, None, None

class DWT2DForward(nn.Module):
    def __init__(self, J=1, wave='db1', mode='zero'):
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
        self.mode = mode

    def forward(self, x):
        yh = []
        ll = x
        m_int = mode_to_int(self.mode)
        for j in range(self.J):
            ll, high = AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, m_int)
            yh.append(high)
        return ll, yh

class DWT2DInverse(nn.Module):
    def __init__(self, wave='db1', mode='zero'):
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
        self.mode = mode

    def forward(self, coeffs):
        yl, yh = coeffs
        ll = yl
        m_int = mode_to_int(self.mode)
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2], ll.shape[-1], device=ll.device)
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            ll = SFB2D.apply(ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row, m_int)
        return ll

def prep_filt_afb3d(h0_x, h1_x, h0_y, h1_y, h0_z, h1_z, device=None):
    """
    Prepares the filters to be of the right form for the afb3d function.
    In particular, makes the tensors the right shape for the
    3D 'analysis filter bank'. This is analogous to prep_filt_afb2d
    but extended to 3D.
    """
    # 1D prep for each pair
    h0_x, h1_x = prep_filt_afb1d(h0_x, h1_x, device=device)  # shape (1,1,L)
    h0_y, h1_y = prep_filt_afb1d(h0_y, h1_y, device=device)  # shape (1,1,L)
    h0_z, h1_z = prep_filt_afb1d(h0_z, h1_z, device=device)  # shape (1,1,L)

    # Now reshape for X dimension (dim=4): (1,1,1,1,L)
    h0_x = h0_x.reshape(1, 1, 1, 1, -1)
    h1_x = h1_x.reshape(1, 1, 1, 1, -1)

    # Reshape for Y dimension (dim=3): (1,1,1,L,1)
    h0_y = h0_y.reshape(1, 1, 1, -1, 1)
    h1_y = h1_y.reshape(1, 1, 1, -1, 1)

    # Reshape for Z dimension (dim=2): (1,1,L,1,1)
    h0_z = h0_z.reshape(1, 1, -1, 1, 1)
    h1_z = h1_z.reshape(1, 1, -1, 1, 1)

    return h0_x, h1_x, h0_y, h1_y, h0_z, h1_z

def afb3d(x, filts, mode='zero'):
    """
    Single-level 3D wavelet decomposition, done by three calls
    to afb1d along the X, Y, and Z dimensions, in that order:
      1) X dimension -> dim=4
      2) Y dimension -> dim=3
      3) Z dimension -> dim=2

    Inputs:
        x (torch.Tensor): shape (N, C, D, H, W)
        filts (list/tuple): (h0_x, h1_x, h0_y, h1_y, h0_z, h1_z) each shaped
                            suitably for afb1d.
        mode (str): padding mode

    Returns:
        y: A tensor of shape (N, C*8, D', H', W') in the 'channels' dimension,
           or, more conveniently, we reshape to (N, C, 8, D', H', W').
    """
    # Unpack filters
    h0_x, h1_x, h0_y, h1_y, h0_z, h1_z = filts

    # 1) Along X dimension = dim=4
    lohi_x = afb1d(x, h0_x, h1_x, mode=mode, dim=4)  # shape (N, 2C, D, H, W/2)

    # 2) Along Y dimension = dim=3
    lohi_xy = afb1d(lohi_x, h0_y, h1_y, mode=mode, dim=3)  # shape (N, 4C, D, H/2, W/2)

    # 3) Along Z dimension = dim=2
    lohi_xyz = afb1d(lohi_xy, h0_z, h1_z, mode=mode, dim=2)  # shape (N, 8C, D/2, H/2, W/2)

    # Reshape so that the 8 subbands are in a separate dimension
    s = lohi_xyz.shape  # (N, 8*C, D', H', W')
    y = lohi_xyz.reshape(s[0], -1, 8, s[-3], s[-2], s[-1])  # (N, C, 8, D', H', W')
    return y

class AFB3D(Function):
    @staticmethod
    def forward(ctx, x, h0_x, h1_x, h0_y, h1_y, h0_z, h1_z, mode):
        """
        Forward pass for one level of 3D wavelet decomposition.
        Returns (low, highs), where highs is size=7 along subband dim.
        """
        mode = int_to_mode(mode)  # convert integer code to string
        ctx.mode = mode
        ctx.save_for_backward(h0_x, h1_x, h0_y, h1_y, h0_z, h1_z)
        ctx.shape = x.shape[-3:]  # original (D, H, W)

        # Apply the 3D analysis
        y = afb3d(x, (h0_x, h1_x, h0_y, h1_y, h0_z, h1_z), mode=mode)
        # y is shape (N, C, 8, D', H', W')
        low = y[:, :, 0].contiguous()       # LLL
        highs = y[:, :, 1:].contiguous()    # The other 7 subbands

        return low, highs

    @staticmethod
    def backward(ctx, dlow, dhighs):
        """
        Backward pass. We do the inverse wavelet transform of
        (dlow, dhighs).
        """
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_x, h1_x, h0_y, h1_y, h0_z, h1_z = ctx.saved_tensors

            # We must call sfb3d to invert:
            # The shape must match the forward usage. We have 1 low subband + 7 high subbands.
            # Recombine them into a single (N, C, 8, D', H', W') for the inverse.
            # dlow: shape (N, C, D', H', W')
            # dhighs: shape (N, C, 7, D', H', W')
            N, C, Dp, Hp, Wp = dlow.shape
            y = torch.cat([dlow[:, :, None], dhighs], dim=2)  # (N, C, 8, D', H', W')

            # Now do the inverse
            dx = sfb3d(y, (h0_x, h1_x, h0_y, h1_y, h0_z, h1_z), mode=mode)
            # dx has shape (N, C, D, H, W) possibly plus 1 if odd dimension, so clip:
            D, H, W = ctx.shape
            dx = dx[:, :, :D, :H, :W]

        # Return grad wrt each input: (x, h0_x, h1_x, h0_y, h1_y, h0_z, h1_z, mode)
        return dx, None, None, None, None, None, None, None

def prep_filt_sfb3d(g0_x, g1_x, g0_y, g1_y, g0_z, g1_z, device=None):
    """
    Prepares the filters to be of the right form for the sfb3d function
    (the 3D 'synthesis filter bank').
    """
    g0_x, g1_x = prep_filt_sfb1d(g0_x, g1_x, device=device)  # shape (1,1,L)
    g0_y, g1_y = prep_filt_sfb1d(g0_y, g1_y, device=device)  # shape (1,1,L)
    g0_z, g1_z = prep_filt_sfb1d(g0_z, g1_z, device=device)  # shape (1,1,L)

    # Reshape for each dimension
    g0_x = g0_x.reshape(1, 1, 1, 1, -1)  # X => dim=4
    g1_x = g1_x.reshape(1, 1, 1, 1, -1)

    g0_y = g0_y.reshape(1, 1, 1, -1, 1)  # Y => dim=3
    g1_y = g1_y.reshape(1, 1, 1, -1, 1)

    g0_z = g0_z.reshape(1, 1, -1, 1, 1)  # Z => dim=2
    g1_z = g1_z.reshape(1, 1, -1, 1, 1)

    return g0_x, g1_x, g0_y, g1_y, g0_z, g1_z

def sfb3d(y, filts, mode='zero'):
    """
    Single-level 3D wavelet reconstruction from subbands y,
    where y is shape (N, C, 8, D', H', W').
    filts = (g0_x, g1_x, g0_y, g1_y, g0_z, g1_z)
    We invert in the reverse order of afb3d:
      1) Z dimension (dim=2)
      2) Y dimension (dim=3)
      3) X dimension (dim=4)
    """
    # Unpack
    g0_x, g1_x, g0_y, g1_y, g0_z, g1_z = filts

    # Split into the 8 subbands: 0 => LLL, 1 => LLH, etc
    # y shape: (N, C, 8, D', H', W')
    lll = y[:, :, 0]
    llh = y[:, :, 1]
    lhl = y[:, :, 2]
    lhh = y[:, :, 3]
    hll = y[:, :, 4]
    hlh = y[:, :, 5]
    hhl = y[:, :, 6]
    hhh = y[:, :, 7]

    # We first merge along Z (dim=2) in pairs:
    # subband0(Lz) + subband1(Hz):
    lo_lz0 = sfb1d(lll[:, :, None, :, :], llh[:, :, None, :, :],
                   g0_z, g1_z, mode=mode, dim=2)
    lo_lz1 = sfb1d(lhl[:, :, None, :, :], lhh[:, :, None, :, :],
                   g0_z, g1_z, mode=mode, dim=2)
    hi_lz0 = sfb1d(hll[:, :, None, :, :], hlh[:, :, None, :, :],
                   g0_z, g1_z, mode=mode, dim=2)
    hi_lz1 = sfb1d(hhl[:, :, None, :, :], hhh[:, :, None, :, :],
                   g0_z, g1_z, mode=mode, dim=2)
    # Now each is shape (N, 2*C, D'', H', W').

    # Next merge along Y (dim=3), grouping pairs that differ in Y:
    lo_y0 = sfb1d(lo_lz0, lo_lz1, g0_y, g1_y, mode=mode, dim=3)
    lo_y1 = sfb1d(hi_lz0, hi_lz1, g0_y, g1_y, mode=mode, dim=3)
    # Now each is shape (N, 4*C, D'', H'', W').

    # Finally merge along X (dim=4):
    x = sfb1d(lo_y0, lo_y1, g0_x, g1_x, mode=mode, dim=4)
    # shape => (N, 8*C, D'', H'', W'')

    # Return final shape (N, C*8, D'', H'', W''), or typically (N, C, D, H, W)
    return x

class SFB3D(Function):
    @staticmethod
    def forward(ctx, low, highs, g0_x, g1_x, g0_y, g1_y, g0_z, g1_z, mode):
        """
        Synthesis of one level in 3D. 
        low:   (N, C, D', H', W')
        highs: (N, C, 7, D', H', W')
        """
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(g0_x, g1_x, g0_y, g1_y, g0_z, g1_z)

        # Recombine into shape (N, C, 8, D', H', W')
        y = torch.cat([low[:, :, None], highs], dim=2)
        x = sfb3d(y, (g0_x, g1_x, g0_y, g1_y, g0_z, g1_z), mode=mode)
        return x

    @staticmethod
    def backward(ctx, dx):
        dlow, dhighs = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            g0_x, g1_x, g0_y, g1_y, g0_z, g1_z = ctx.saved_tensors

            # Use afb3d to get partial derivatives wrt each subband
            # dx shape: (N, C, D, H, W) or bigger if needed
            y = afb3d(dx, (g0_x, g1_x, g0_y, g1_y, g0_z, g1_z), mode=mode)
            # y shape => (N, C, 8, D', H', W')

            # The first subband is "low", the next 7 are "highs"
            dlow = y[:, :, 0].contiguous()
            dhighs = y[:, :, 1:].contiguous()

        # Return grads for (low, highs, g0_x, g1_x, g0_y, g1_y, g0_z, g1_z, mode)
        return dlow, dhighs, None, None, None, None, None, None, None

class DWT3DForward(nn.Module):
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            # Same filters for x, y, z
            h0, h1 = wave.dec_lo, wave.dec_hi
            h0_x, h1_x = h0, h1
            h0_y, h1_y = h0, h1
            h0_z, h1_z = h0, h1
        else:
            # Suppose wave = [h0, h1] or [h0_x,h1_x,h0_y,h1_y,h0_z,h1_z]
            # as needed. For simplicity, if len=2, reuse for all dims
            # If len=6, pick them out
            if len(wave) == 2:
                h0_x, h1_x = wave
                h0_y, h1_y = wave
                h0_z, h1_z = wave
            elif len(wave) == 6:
                h0_x, h1_x, h0_y, h1_y, h0_z, h1_z = wave

        filts = prep_filt_afb3d(h0_x, h1_x, h0_y, h1_y, h0_z, h1_z)
        self.register_buffer('h0_x', filts[0])
        self.register_buffer('h1_x', filts[1])
        self.register_buffer('h0_y', filts[2])
        self.register_buffer('h1_y', filts[3])
        self.register_buffer('h0_z', filts[4])
        self.register_buffer('h1_z', filts[5])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """
        x: (N, C, D, H, W)
        Returns (yl, [yh1, yh2, ..., yhJ])
         where yl is the lowpass after J scales,
         and each yh is shape (N, C, 7, D', H', W') for that scale.
        """
        yh = []
        ll = x
        m_int = mode_to_int(self.mode)
        for j in range(self.J):
            ll, high = AFB3D.apply(ll, self.h0_x, self.h1_x,
                                       self.h0_y, self.h1_y,
                                       self.h0_z, self.h1_z, m_int)
            yh.append(high)
        return ll, yh

class DWT3DInverse(nn.Module):
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, g1 = wave.rec_lo, wave.rec_hi
            g0_x, g1_x = g0, g1
            g0_y, g1_y = g0, g1
            g0_z, g1_z = g0, g1
        else:
            # If wave is custom, parse similarly as above
            if len(wave) == 2:
                g0_x, g1_x = wave
                g0_y, g1_y = wave
                g0_z, g1_z = wave
            elif len(wave) == 6:
                g0_x, g1_x, g0_y, g1_y, g0_z, g1_z = wave

        filts = prep_filt_sfb3d(g0_x, g1_x, g0_y, g1_y, g0_z, g1_z)
        self.register_buffer('g0_x', filts[0])
        self.register_buffer('g1_x', filts[1])
        self.register_buffer('g0_y', filts[2])
        self.register_buffer('g1_y', filts[3])
        self.register_buffer('g0_z', filts[4])
        self.register_buffer('g1_z', filts[5])
        self.mode = mode

    def forward(self, coeffs):
        """
        coeffs = (yl, [yh1, yh2, ... yhJ])
        Each yh is (N, C, 7, D', H', W')
        """
        yl, yh = coeffs
        m_int = mode_to_int(self.mode)
        out = yl
        # Reconstruct from the top scale down
        for h in reversed(yh):
            if h is None:
                # Just fill with zeros if needed
                h = torch.zeros(
                    out.shape[0], out.shape[1], 7, out.shape[-3], out.shape[-2], out.shape[-1],
                    device=out.device, dtype=out.dtype
                )
            out = SFB3D.apply(out, h,
                              self.g0_x, self.g1_x,
                              self.g0_y, self.g1_y,
                              self.g0_z, self.g1_z,
                              m_int)
        return out
