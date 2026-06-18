from torch import nn

from src.model.le_admm import LeADMM


class ModularLeADMM(nn.Module):
    def __init__(
        self,
        mu1=1e-4,
        mu2=1e-4,
        mu3=1e-4,
        tau=2e-4,
        leadmm_iters=5,
        pad_factor=2,
        preprocess=None,
        postprocess=None,
    ):
        super().__init__()
        self.preprocess = preprocess
        self.reconstructor = LeADMM(mu1, mu2, mu3, tau, leadmm_iters, pad_factor)
        self.postprocess = postprocess

    def forward(self, lensless, psf, **batch):
        if self.preprocess is not None:
            lensless = self.preprocess(lensless)

        admm_pred = self.reconstructor(lensless, psf)['prediction']
        pred = admm_pred

        if self.postprocess is not None:
            pred = self.postprocess(pred)

        return {
            "prediction": pred,
            "admm_prediction": admm_pred,
        }
