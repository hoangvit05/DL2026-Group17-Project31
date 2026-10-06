import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DoubleConv(nn.Module):
    """
    Standard Double Convolution block with BatchNorm and ReLU.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class VanillaUNet(nn.Module):
    """
    Classical Vanilla U-Net (4-stage resolution encoder-decoder) trained from scratch.
    """
    def __init__(self, in_channels=3, classes=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 4, base * 8))
        self.d4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 8, base * 16))

        self.up1 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.c1 = DoubleConv(base * 16, base * 8)
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.c2 = DoubleConv(base * 8, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.c3 = DoubleConv(base * 4, base * 2)
        self.up4 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.c4 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.c1(torch.cat([x4, self.up1(x5)], dim=1))
        x = self.c2(torch.cat([x3, self.up2(x)], dim=1))
        x = self.c3(torch.cat([x2, self.up3(x)], dim=1))
        x = self.c4(torch.cat([x1, self.up4(x)], dim=1))
        return self.outc(x)


def build_model(model_name='unet', encoder='resnet34', in_channels=3, classes=1, pretrained=True):
    """
    Model factory: returns either Vanilla U-Net or Pretrained SMP U-Net.
    """
    if model_name.lower() in ['vanilla_unet', 'vanilla']:
        return VanillaUNet(in_channels=in_channels, classes=classes)
    weights = 'imagenet' if pretrained else None
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes
    )
