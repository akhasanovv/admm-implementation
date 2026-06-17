from torch import nn


class ADMM(nn.Module):
    """
    ADMM implementation from [paper](https://arxiv.org/pdf/1908.11502)
    """

    def __init__(self, mu1=1e-4, mu2=1e-4, mu3=1e-4, tau=2e-4):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.tau = tau

    def forward(self, lensless, psf, num_iters=100, **batch):
        b = lensless
        for _ in range(num_iters):
            pass

        return {"prediction": b}

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
