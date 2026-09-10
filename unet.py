import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Dois blocos consecutivos de:
    Conv2d -> ReLU -> Conv2d -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net para segmentação semântica ou segmentação
    baseada em interior/borda.

    Parâmetros:
        in_channels: número de canais da imagem de entrada.
        num_classes: número de canais/classes na saída.
        base_channels: número inicial de filtros.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        base_channels=32
    ):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(
            in_channels,
            base_channels
        )

        self.enc2 = DoubleConv(
            base_channels,
            base_channels * 2
        )

        self.enc3 = DoubleConv(
            base_channels * 2,
            base_channels * 4
        )

        self.enc4 = DoubleConv(
            base_channels * 4,
            base_channels * 8
        )

        # Bottleneck
        self.bottleneck = DoubleConv(
            base_channels * 8,
            base_channels * 16
        )

        # Operações de downsampling
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            base_channels * 16,
            base_channels * 8,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            base_channels * 16,
            base_channels * 8
        )

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            base_channels * 8,
            base_channels * 4
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            base_channels * 4,
            base_channels * 2
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            base_channels * 2,
            base_channels
        )

        # Camada final
        self.out = nn.Conv2d(
            base_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        u4 = self.up4(b)
        d4 = self.dec4(
            torch.cat([u4, e4], dim=1)
        )

        u3 = self.up3(d4)
        d3 = self.dec3(
            torch.cat([u3, e3], dim=1)
        )

        u2 = self.up2(d3)
        d2 = self.dec2(
            torch.cat([u2, e2], dim=1)
        )

        u1 = self.up1(d2)
        d1 = self.dec1(
            torch.cat([u1, e1], dim=1)
        )

        # Saída
        return self.out(d1)