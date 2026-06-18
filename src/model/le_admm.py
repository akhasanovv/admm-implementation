import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from src.model.admm import ADMM

class LeADMM(ADMM):
    """
    Le-ADMM from [paper](https://arxiv.org/pdf/1908.11502)
    """

    def __init__(
        self, 
        mu1=1e-4,
        mu2=1e-4,
        mu3=1e-4,
        tau=2e-4,
        num_iters=20,
        pad_factor=2,
    ):
        super().__init__(mu1, mu2, mu3, tau, pad_factor)
        
        self.num_iters = num_iters
        self.mu1_init = nn.Parameter(torch.full((num_iters,), self._get_positive_inv(mu1)))
        self.mu2_init = nn.Parameter(torch.full((num_iters,), self._get_positive_inv(mu2)))
        self.mu3_init = nn.Parameter(torch.full((num_iters,), self._get_positive_inv(mu3)))
        self.tau_init = nn.Parameter(torch.full((num_iters,), self._get_positive_inv(tau)))
    
    def _get_positive(self, param): # F(x) = log(1 + exp(x))
        return F.softplus(param)
    
    def _get_positive_inv(self, param): # F^{-1}(x) = log(exp(x) - 1)
        return np.log(np.exp(param) - 1) # -> float
    
    def forward(self, lensless, psf, **batch):
        shape = self._padded_shape(lensless, psf)
        h_fft = self._psf_fft(psf, shape)
        ct_c = self.c_t_c(lensless, shape)
        b_pad = self._embed(lensless, shape)
        
        x = torch.zeros_like(b_pad)
        u = torch.zeros(*x.shape[:2], 2, *x.shape[-2:], device=x.device, dtype=x.dtype)
        v = torch.zeros_like(x)
        w = torch.zeros_like(x)
        a1 = torch.zeros_like(x)
        a2 = torch.zeros_like(u)
        a3 = torch.zeros_like(x)

        for i in range(self.num_iters):
            mu1 = self._get_positive(self.mu1_init[i])
            mu2 = self._get_positive(self.mu2_init[i])
            mu3 = self._get_positive(self.mu3_init[i])
            tau = self._get_positive(self.tau_init[i])
            
            u = self.T(self.psi(x) + a2 / mu2, tau / mu2)
            v = (a1 + mu1 * self.fft_mul(x, h_fft) + b_pad) / (ct_c + mu1)
            w = torch.max(a3 / mu3 + x, torch.zeros_like(x))
            
            denom = mu1 * h_fft.abs().square() + mu2 * self.psi_t_psi(shape, lensless.device, lensless.dtype) + mu3

            r = mu3 * w - a3
            r = r + self.psi_t(mu2 * u - a2)
            r = r + self.fft_mul_t(mu1 * v - a1, h_fft)
            x = torch.fft.ifft2(torch.fft.fft2(r) / denom).real

            hx = self.fft_mul(x, h_fft)
            gx = self.psi(x)
            a1 = a1 + mu1 * (hx - v)
            a2 = a2 + mu2 * (gx - u)
            a3 = a3 + mu3 * (x - w)

        pred = self._crop(x, lensless.shape[-2:])
        return {"prediction": pred}
