import torch
import torch.nn as nn

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
        nn.InstanceNorm3d(out_c, affine=True),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
        nn.InstanceNorm3d(out_c, affine=True),
        nn.LeakyReLU(0.1, inplace=True),
    )
def _crop_to(x, ref):
    """Center-crop x spatially to match ref's D,H,W."""
    dz = x.size(2) - ref.size(2)
    dy = x.size(3) - ref.size(3)
    dx = x.size(4) - ref.size(4)
    z0 = dz // 2; y0 = dy // 2; x0 = dx // 2
    z1 = x.size(2) - (dz - z0)
    y1 = x.size(3) - (dy - y0)
    x1 = x.size(4) - (dx - x0)
    return x[:, :, z0:z1, y0:y1, x0:x1]
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, base=32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base*2, base*4, base*8, base*16

        self.enc1 = conv_block(in_channels, c1)
        self.down1 = nn.Conv3d(c1, c2, 2, stride=2)
        self.enc2 = conv_block(c2, c2)
        self.down2 = nn.Conv3d(c2, c3, 2, stride=2)
        self.enc3 = conv_block(c3, c3)
        self.down3 = nn.Conv3d(c3, c4, 2, stride=2)
        self.enc4 = conv_block(c4, c4)
        self.down4 = nn.Conv3d(c4, c5, 2, stride=2)

        self.bottleneck = conv_block(c5, c5)

        self.up4 = nn.ConvTranspose3d(c5, c4, 2, stride=2)
        self.dec4 = conv_block(c4 + c4, c4)
        self.up3 = nn.ConvTranspose3d(c4, c3, 2, stride=2)
        self.dec3 = conv_block(c3 + c3, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = conv_block(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = conv_block(c1 + c1, c1)

        self.out = nn.Conv3d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b  = self.bottleneck(self.down4(e4))

        u4 = self.up4(b)
        e4m = _crop_to(e4, u4) if e4.shape != u4.shape else e4
        d4 = self.dec4(torch.cat([u4, e4m], dim=1))

        u3 = self.up3(d4)
        e3m = _crop_to(e3, u3) if e3.shape != u3.shape else e3
        d3 = self.dec3(torch.cat([u3, e3m], dim=1))

        u2 = self.up2(d3)
        e2m = _crop_to(e2, u2) if e2.shape != u2.shape else e2
        d2 = self.dec2(torch.cat([u2, e2m], dim=1))

        u1 = self.up1(d2)
        e1m = _crop_to(e1, u1) if e1.shape != u1.shape else e1
        d1 = self.dec1(torch.cat([u1, e1m], dim=1))

        return torch.sigmoid(self.out(d1))
