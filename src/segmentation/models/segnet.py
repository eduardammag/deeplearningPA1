import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Dois blocos convolucionais 3x3 com ReLU.
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


class SegNet(nn.Module):
    """
    SegNet simplificada para a ablação de recuperação de resolução.

    O encoder utiliza MaxPooling com índices.
    O decoder recupera a resolução utilizando MaxUnpool2d.

    Diferentemente da U-Net, não são utilizadas skip connections.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        base_channels=32
    ):
        super().__init__()

        # ==================================================
        # Encoder
        # ==================================================

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

        # ==================================================
        # Bottleneck
        #
        # Mantemos 256 canais para que sejam compatíveis
        # com os índices produzidos pelo último MaxPool.
        # ==================================================

        self.bottleneck = DoubleConv(
            base_channels * 8,
            base_channels * 8
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=True
        )

        # ==================================================
        # Decoder
        # ==================================================

        self.unpool = nn.MaxUnpool2d(
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            base_channels * 8,
            base_channels * 4
        )

        self.dec3 = DoubleConv(
            base_channels * 4,
            base_channels * 2
        )

        self.dec2 = DoubleConv(
            base_channels * 2,
            base_channels
        )

        self.dec1 = DoubleConv(
            base_channels,
            base_channels
        )

        # ==================================================
        # Output
        # ==================================================

        self.out = nn.Conv2d(
            base_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # ==================================================
        # Encoder
        # ==================================================

        e1 = self.enc1(x)

        p1, indices1 = self.pool(e1)

        e2 = self.enc2(p1)

        p2, indices2 = self.pool(e2)

        e3 = self.enc3(p2)

        p3, indices3 = self.pool(e3)

        e4 = self.enc4(p3)

        p4, indices4 = self.pool(e4)

        # ==================================================
        # Bottleneck
        # ==================================================

        b = self.bottleneck(p4)

        # ==================================================
        # Decoder
        # ==================================================

        d4 = self.unpool(
            b,
            indices4,
            output_size=e4.size()
        )

        d4 = self.dec4(d4)

        d3 = self.unpool(
            d4,
            indices3,
            output_size=e3.size()
        )

        d3 = self.dec3(d3)

        d2 = self.unpool(
            d3,
            indices2,
            output_size=e2.size()
        )

        d2 = self.dec2(d2)

        d1 = self.unpool(
            d2,
            indices1,
            output_size=e1.size()
        )

        d1 = self.dec1(d1)

        return self.out(d1)