import imp
from matplotlib.pyplot import text

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from .transformer import *
from .position_encoding import *
from .mask_decoder import *
from .conv_lstm import ConvLSTM
from timesformer.models.vit import TimeSformer

class ConvLSTMBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, vision_encoder, hidden_dim=768, mask_dim=112, traj_dim=56, spatial_dim=14, num_frames=16, attn_type="dot_product"):
        super(ConvLSTMBaseline, self).__init__()

        self.spatial_dim = spatial_dim
        self.num_frames = num_frames
        
        self.attn_type = attn_type

        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)
        self.sub_text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        self.conv3d = nn.Conv3d(192, hidden_dim, kernel_size=3, stride=1, padding=1)
        
        self.bilinear = nn.Bilinear(self.num_frames * 49, 20, self.num_frames * 49)
        
        self.mm_decoder = ConvLSTM(
            input_dim=hidden_dim,
            mask_dim=mask_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            return_all_layers=False,
            attn_type=self.attn_type
        )

        self.traj_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 4, 6, 8], out_channels=256),
            ConvUpsample(in_channels=256,
                         out_channels=1,
                         channels=[256, 256, 128],
                         upsample=[True, True, True],
                         drop=0.2,
                         ),
            nn.Upsample(
                size=(traj_dim, traj_dim), mode="bilinear", align_corners=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, frames, sub_text, frame_mask, sub_text_mask):

        bs = frames.shape[0]
        nf = self.num_frames
        
        vision_feat = self.vision_encoder(frames)
        vision_feat = F.relu(self.conv3d(vision_feat))
        
        sub_text = rearrange(sub_text, "b n l c -> (b n) l c")
        sub_text_feat = self.sub_text_encoder(sub_text)
        sub_text_feat = rearrange(sub_text_feat, "(b n) l c -> b n l c", b=bs, n=nf)

        hidden_feat, segm_mask = self.mm_decoder(vision_feat, sub_text_feat, frame_mask, sub_text_mask)  # .squeeze(1)
        
        # use last hidden state
        traj_mask = self.traj_decoder(hidden_feat[-1][0])

        return segm_mask, traj_mask


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
