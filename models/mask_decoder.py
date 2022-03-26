import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoUpsample(nn.Module):
    def __init__(
        self,
        in_channels=512,
        out_channels=2,
        channels=[512, 256, 128, 64],
        upsample=[True, True, False, False],
        scale_factor=2,
        drop=0.2,
    ):
        super().__init__()
        
        linear_upsampling = nn.Upsample(scale_factor=(1, scale_factor, scale_factor), mode='trilinear')
        
        assert len(channels) == len(upsample)
        
        modules = []

        for i in range(len(channels)):

            modules.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        channels[i],
                        kernel_size=(1, 3, 3),
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    # nn.BatchNorm3d(channels[i]),
                    nn.ReLU(),
                    nn.Dropout3d(drop),
                )
            )

            if upsample[i]:
                modules.append(linear_upsampling)

            in_channels = channels[i]

        modules.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 3, 3),
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )
        )

        self.deconv = nn.Sequential(*modules)

    def forward(self, x):
        return self.deconv(x)
        


class ConvUpsample(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        out_channels=1,
        channels=[512, 256, 128, 64],
        upsample=[True, True, False, False],
        scale_factor=2,
        drop=0.2,
    ):
        super().__init__()

        linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        assert len(channels) == len(upsample)

        modules = []

        for i in range(len(channels)):

            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    # nn.BatchNorm2d(channels[i]),
                    nn.ReLU(),
                    nn.Dropout2d(drop),
                )
            )

            if upsample[i]:
                modules.append(linear_upsampling)

            in_channels = channels[i]

        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )
        )

        self.deconv = nn.Sequential(*modules)

    def forward(self, x):
        return self.deconv(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
