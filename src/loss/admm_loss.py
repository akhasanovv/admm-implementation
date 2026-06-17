import torch
from torch import nn
from torch.nn import functional as F


class ReconstructionLoss(nn.Module):
    """
    reconstruction loss from [paper](https://arxiv.org/pdf/1908.11502)
    """

    def __init__(self, mse_weight=1.0, lpips_weight=0.0, lpips_net="vgg"):
        super().__init__()
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.lpips = None

        if lpips_weight > 0:
            import lpips

            self.lpips = lpips.LPIPS(net=lpips_net)

    def forward(self, prediction: torch.Tensor, lensed: torch.Tensor, **batch):
        """
        calculate loss

        Args:
            prediction (Tensor): reconstructed images in [0, 1]
            lensed (Tensor): ground-truth images in [0, 1]

        Returns:
            float: loss
        """
        prediction = prediction.clamp(0, 1)
        lensed = lensed.clamp(0, 1)

        mse_loss = F.mse_loss(prediction, lensed)
        loss = self.mse_weight * mse_loss
        result = {"loss": loss, "mse_loss": mse_loss}

        if self.lpips is not None:
            lpips_loss = self.lpips(
                prediction * 2 - 1,
                lensed * 2 - 1,
            ).mean()
            result["lpips_loss"] = lpips_loss
            result["loss"] = result["loss"] + self.lpips_weight * lpips_loss

        return result


ADMMLoss = ReconstructionLoss
