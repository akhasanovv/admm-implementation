import torch
from torch import nn
from torch.nn import functional as F


class ADMM(nn.Module):
    """
    ADMM from [paper](https://arxiv.org/pdf/1908.11502). no trainable params. only math only hardcore
    """
    def __init__(
        self,
        mu1=1e-4,
        mu2=1e-4,
        mu3=1e-4,
        tau=2e-4,
        pad_factor=2,
    ):
        super().__init__()
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.tau = tau
        self.pad_factor = pad_factor

    def forward(self, lensless, psf, num_iters=100, **batch):
        shape = self._padded_shape(lensless, psf)
        h_fft = self._psf_fft(psf, shape)
        ct_c = self.c_t_c(lensless, shape)
        b_pad = self._embed(lensless, shape)

        mu1, mu2, mu3, tau = self.mu1, self.mu2, self.mu3, self.tau
        denom = mu1 * h_fft.abs().square() + mu2 * self.psi_t_psi(shape, lensless.device, lensless.dtype) + mu3

        x = torch.zeros_like(b_pad)
        u = torch.zeros(*x.shape[:2], 2, *x.shape[-2:], device=x.device, dtype=x.dtype)
        v = torch.zeros_like(x)
        w = torch.zeros_like(x)
        a1 = torch.zeros_like(x)
        a2 = torch.zeros_like(u)
        a3 = torch.zeros_like(x)

        for _ in range(num_iters):
            u = self.T(self.psi(x) + a2 / mu2, tau / mu2)
            v = (a1 + mu1 * self.fft_mul(x, h_fft) + b_pad) / (ct_c + mu1)
            w = torch.max(a3 / mu3 + x, torch.zeros_like(x))

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
        return {"prediction": pred.clamp(0, 1)}

    def _padded_shape(self, lensless, psf):
        h, w = lensless.shape[-2:]
        ph = max(self.pad_factor * h, psf.shape[-2])
        pw = max(self.pad_factor * w, psf.shape[-1])
        return ph, pw

    def _psf_fft(self, psf, shape):
        psf = self._center_pad_or_crop(psf, shape)
        psf = torch.fft.ifftshift(psf, dim=(-2, -1))
        return torch.fft.fft2(psf) # -> fft(h)

    def _center_pad_or_crop(self, x, shape):
        h, w = x.shape[-2:]
        th, tw = shape
        top = max((h - th) // 2, 0)
        left = max((w - tw) // 2, 0)
        x = x[..., top : top + min(h, th), left : left + min(w, tw)]

        pad_h = th - x.shape[-2]
        pad_w = tw - x.shape[-1]
        return F.pad(
            x,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
        )

    def _crop(self, x, shape):
        h, w = shape
        top = (x.shape[-2] - h) // 2
        left = (x.shape[-1] - w) // 2
        return x[..., top : top + h, left : left + w]

    def _embed(self, x, shape):
        return self._center_pad_or_crop(x, shape)

    def c_t_c(self, lensless, shape): # C^T C
        mask = torch.zeros(
            *lensless.shape[:2],
            *shape,
            device=lensless.device,
            dtype=lensless.dtype,
        )
        rows, cols = self._crop_slice(shape, lensless.shape[-2:])
        mask[..., rows, cols] = 1
        return mask

    def _crop_slice(self, shape, crop_shape):
        h, w = crop_shape
        top = (shape[0] - h) // 2
        left = (shape[1] - w) // 2
        return slice(top, top + h), slice(left, left + w)

    def fft_mul(self, x, h_fft): # H x = h*x = F^{-1}(F(h) \cdot F(x))
        return torch.fft.ifft2(torch.fft.fft2(x) * h_fft).real

    def fft_mul_t(self, x, h_fft): # H^T x = F^{-1}(\overline{F(h)} \cdot F(x))
        return torch.fft.ifft2(torch.fft.fft2(x) * h_fft.conj()).real

    def psi(self, x):
        dx = torch.roll(x, shifts=-1, dims=-1) - x
        dy = torch.roll(x, shifts=-1, dims=-2) - x
        return torch.stack((dx, dy), dim=2)

    def psi_t(self, grad):
        dx, dy = grad[:, :, 0], grad[:, :, 1]
        return (
            torch.roll(dx, shifts=1, dims=-1)
            - dx
            + torch.roll(dy, shifts=1, dims=-2)
            - dy
        )

    def T(self, x, threshold): # soft thresholding
        return torch.sign(x) * torch.relu(x.abs() - threshold)

    def psi_t_psi(self, shape, device, dtype):
        h, w = shape
        fy = torch.fft.fftfreq(h, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(w, device=device, dtype=dtype)
        den_y = 2 - 2 * torch.cos(2 * torch.pi * fy)
        den_x = 2 - 2 * torch.cos(2 * torch.pi * fx)
        return den_y[:, None] + den_x[None, :]

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
