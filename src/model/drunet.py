from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class DRUNet(nn.Module):
    def __init__(self, channels, in_channels=3, out_channels=3, blocks_per_scale=4):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.head = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.enc1 = self._blocks(c1, blocks_per_scale)
        self.down1 = nn.Conv2d(c1, c2, 2, stride=2)
        self.enc2 = self._blocks(c2, blocks_per_scale)
        self.down2 = nn.Conv2d(c2, c3, 2, stride=2)
        self.enc3 = self._blocks(c3, blocks_per_scale)
        self.down3 = nn.Conv2d(c3, c4, 2, stride=2)

        self.mid = self._blocks(c4, blocks_per_scale)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = self._blocks(c3, blocks_per_scale)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = self._blocks(c2, blocks_per_scale)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = self._blocks(c1, blocks_per_scale)
        self.tail = nn.Conv2d(c1, out_channels, 3, padding=1)

    def forward(self, x):
        y1 = self.enc1(F.relu(self.head(x)))
        y2 = self.enc2(F.relu(self.down1(y1)))
        y3 = self.enc3(F.relu(self.down2(y2)))
        y = self.mid(F.relu(self.down3(y3)))

        y = self.dec3(self._match(self.up3(y), y3) + y3)
        y = self.dec2(self._match(self.up2(y), y2) + y2)
        y = self.dec1(self._match(self.up1(y), y1) + y1)
        return (x + self.tail(y)).clamp(0, 1)

    def _blocks(self, channels, count):
        return nn.Sequential(*[ResBlock(channels) for _ in range(count)])

    def _match(self, x, target):
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear")
