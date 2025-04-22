import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)
        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)

        self.bottleneck = conv_block(32, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = conv_block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = conv_block(32, 16)

        self.final = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        
        up2 = self.up2(bottleneck)
        if up2.shape[-2:] != enc2.shape[-2:]:
            up2 = T.functional.resize(up2, size=enc2.shape[-2:])
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.up1(dec2)
        if up1.shape[-2:] != enc1.shape[-2:]:
            up1 = T.functional.resize(up1, size=enc1.shape[-2:])
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        
        return self.final(dec1)
    
    def __str__(self):
        return f"UNet Model Structure:\n" \
            f"- Input Channels: {self.enc1[0].in_channels}\n" \
            f"- Output Channels: {self.final.out_channels}\n" \
            f"- Encoder: 16 → 32 → Bottleneck 64\n" \
            f"- Decoder: 64 → 32 → 16 → Output"

    def __repr__(self):
        return f"UNet(in_channels={self.enc1[0].in_channels}, out_channels={self.final.out_channels})"

# model = UNet()
# print(model)            # 會呼叫 __str__()
# print(repr(model))      # 明確呼叫 __repr__()
