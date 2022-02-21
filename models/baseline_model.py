from matplotlib.pyplot import text

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .mask_decoder import *


# simplest thing should be to predict a segmentation mask first
class SegmentationBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, vision_encoder, mask_dim=112):
        super(SegmentationBaseline, self).__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=384)

        # self.mm_fusion = None

        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=2 * 384, atrous_rates=[6, 12, 24], out_channels=256),
            ConvUpsample(
                in_channels=256,
                out_channels=1,
                channels=[256, 256, 128],
                upsample=[True, True, True],
                drop=0.2,
            ),
            nn.Upsample(size=(mask_dim, mask_dim), mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )

    def forward(self, frames, text, text_mask):

        vision_feat = self.vision_encoder(frames)
        vision_feat = F.normalize(vision_feat, p=2, dim=1)  # B x N x C
        vision_dim = int(vision_feat.shape[1] ** 0.5)
        vision_feat = rearrange(
            vision_feat, "b (h w) c -> b c h w", h=vision_dim, w=vision_dim
        )

        text_feat = self.text_encoder(text)  # B x L x C
        text_feat = F.normalize(text_feat, p=2, dim=1)  # B x L x C
        text_feat = text_feat * text_mask[:, :, None]
        text_feat = torch.mean(text_feat, dim=1)
        text_feat = text_feat.repeat(vision_dim, vision_dim, 1, 1).view(
            *vision_feat.shape
        )

        # import pdb; pdb.set_trace()
        fused_feat = torch.cat((vision_feat, text_feat), 1)
        segm_mask = self.mm_decoder(fused_feat)

        return segm_mask


# TODO
class FullBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, vision_encoder, text_encoder):
        super(FullBaseline, self).__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        self.mm_fusion = None
        self.mm_decoder = None

        self.destination_extractor = None
        self.position_net = None

    # position - curr vehcle pos 3d -> (proj trans) 2d curr pos
    # next vehicle pos 3d -> (proj trans) 2d next pos
    # offset - 2d offset
    # in eval - 2d to 3d inverse proj trans
    def forward(self, frames, text, position):

        vision_feat = self.vision_encoder(frames)
        text_feat = self.text_encoder(text)

        fused_feat = self.mm_fusion(vision_feat, text_feat)
        segm_mask = self.mm_decoder(fused_feat)

        destination = self.destination_extractor(segm_mask)
        position_offset = self.position_net(position, destination)

        return segm_mask, destination, position_offset


class TextEncoder(nn.Module):
    def __init__(
        self,
        input_size=300,
        hidden_size=192,
        num_layers=1,
        batch_first=True,
        dropout=0.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, input):
        self.lstm.flatten_parameters()
        output, (_, _) = self.lstm(input)
        return output
