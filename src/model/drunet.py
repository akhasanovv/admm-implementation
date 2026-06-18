import torch
from torch import nn
from torch.nn import functional as F


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.net = nn.Sequential(
            ConvReLU(in_channels, out_channels),
            ConvReLU(out_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.net = nn.Sequential(
            ConvReLU(in_channels, out_channels),
            ConvReLU(out_channels, out_channels),
            ConvReLU(out_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class DRUNet(nn.Module):
    """
    UNet as described in [paper](https://arxiv.org/pdf/1908.11502)
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        self.enc1 = EncBlock(in_channels, 24)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = EncBlock(24, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = EncBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = EncBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.enc5 = EncBlock(256, 512)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv1 = ConvReLU(512, 512)
        
        self.dec5 = DecBlock(1024, 256)
        self.dec4 = DecBlock(512, 128)
        self.dec3 = DecBlock(256, 64)
        self.dec2 = DecBlock(128, 24)
        self.dec1 = DecBlock(48, 24)
        self.conv2 = nn.Conv2d(24, out_channels, 1)

    def _concat(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear")
        return torch.cat((x, skip), dim=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        enc5 = self.enc5(self.pool4(enc4))

        y = self.conv1(self.pool5(enc5))
        y = self.dec5(self._concat(y, enc5))
        y = self.dec4(self._concat(y, enc4))
        y = self.dec3(self._concat(y, enc3))
        y = self.dec2(self._concat(y, enc2))
        y = self.dec1(self._concat(y, enc1))
        
        return self.conv2(y)
