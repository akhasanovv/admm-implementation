import torch
from torch.nn import functional as F

from src.metrics.base_metric import BaseMetric

import lpips

class ReconstructionMetric(BaseMetric):
    def __init__(self, metric, device="auto", clamp=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)
        self.clamp = clamp

    def __call__(self, prediction: torch.Tensor, lensed: torch.Tensor, **kwargs):
        prediction = prediction.clamp(0, 1) if self.clamp else prediction
        lensed = lensed.clamp(0, 1) if self.clamp else lensed
        return self.metric(prediction, lensed).item()


class MSEMetric(BaseMetric):
    def __call__(self, prediction: torch.Tensor, lensed: torch.Tensor, **kwargs):
        return F.mse_loss(prediction.clamp(0, 1), lensed.clamp(0, 1)).item()


class LPIPSMetric(BaseMetric):
    def __init__(self, net="vgg", device="auto", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = lpips.LPIPS(net=net).to(device)

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, lensed: torch.Tensor, **kwargs):
        prediction = prediction.clamp(0, 1) * 2 - 1
        lensed = lensed.clamp(0, 1) * 2 - 1
        return self.metric(prediction, lensed).mean().item()
