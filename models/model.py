import imp
from matplotlib.pyplot import text

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from .transformer import *
from .position_encoding import *
from .mask_decoder import *
from timesformer.models.vit import TimeSformer


# simplest thing should be to predict a segmentation mask first
class SegmentationBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, vision_encoder, hidden_dim=384, mask_dim=112, backbone="vit"):
        super(SegmentationBaseline, self).__init__()

        self.backbone = backbone
        
        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        # self.mm_fusion = None

        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[6, 12, 24], out_channels=256),
            ConvUpsample(in_channels=256,
                         out_channels=1,
                         channels=[256, 256, 128],
                         upsample=[True, True, True],
                         drop=0.2,
                         ),
            nn.Upsample(
                size=(mask_dim, mask_dim), mode="bilinear", align_corners=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, frames, text, frame_mask, text_mask):

        vision_feat = self.vision_encoder(frames)
        if self.backbone.startswith('deeplabv3_'):
            vision_feat = rearrange(vision_feat, "b c h w -> b (h w) c")

        vision_feat = F.normalize(vision_feat, p=2, dim=1)  # B x N x C
        vision_dim = int(vision_feat.shape[1]**.5)

        # vision_feat = rearrange(vision_feat, "b (h w) c -> b c h w", h=14, w=14)

        text_feat = self.text_encoder(text)  # B x L x C
        text_feat = F.normalize(text_feat, p=2, dim=1)  # B x L x C
        text_feat = text_feat * text_mask[:, :, None]

        # import pdb; pdb.set_trace()
        # print(vision_feat.shape, text_feat.shape)
        cross_attn = torch.bmm(vision_feat, text_feat.transpose(1, 2).contiguous())  # B x N x L
        cross_attn = cross_attn.softmax(dim=-1)
        attn_feat = cross_attn @ text_feat  # B x N x C

        fused_feat = vision_feat * attn_feat
        
        fused_feat = rearrange(fused_feat, "b (h w) c -> b c h w", h=vision_dim, w=vision_dim)

        segm_mask = self.mm_decoder(fused_feat)  # .squeeze(1)

        return segm_mask

class VideoSegmentationBaseline(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, vision_encoder, hidden_dim=768, mask_dim=112):
        super(VideoSegmentationBaseline, self).__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)
        
        self.mm_fusion = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, None, None))
        )
        
        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[6, 12, 24], out_channels=256),
            ConvUpsample(in_channels=256,
                out_channels=1,
                channels=[256, 256, 128],
                upsample=[True, True, True],
                drop=0.2,
            ),
            nn.Upsample(
                size=(mask_dim, mask_dim), mode="bilinear", align_corners=True
            ),
            nn.Sigmoid(),
        )  

    def forward(self, frames, text, frame_mask, text_mask):
        
        vision_feat, _ = self.vision_encoder(frames) # B, N, C
        vision_feat = F.normalize(vision_feat, p=2, dim=1) # B x N x C
        vision_feat = rearrange(vision_feat, "b (n h w) c -> b c n h w", n=8, h=14, w=14)
        
        text_feat = self.text_encoder(text) # B x L x C
        text_feat = F.normalize(text_feat, p=2, dim=1) # B x L x C
        text_feat = text_feat * text_mask[:, :, None]
        text_feat = text_feat.mean(dim=1)
        text_feat = repeat(text_feat, "b c -> b c n h w", n=8, h=14, w=14)
        
        fused_feat = torch.cat([vision_feat, text_feat], dim=1)
        fused_feat = self.mm_fusion(fused_feat)
        fused_feat = rearrange("b c 1 h w -> b c h w")
        
        segm_mask = self.mm_decoder(fused_feat) #.squeeze(1)

        return segm_mask

class IROSBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, vision_encoder, hidden_dim=384, mask_dim=112, num_encoder_layers=2, normalize_before=True):
        super(IROSBaseline, self).__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        # self.frame_mask = torch.ones(1, 14*14, dtype=torch.int64)

        self.pool = nn.AdaptiveMaxPool2d((28, 28))
        self.conv_3x3 = nn.ModuleDict(
            {
                "layer2": nn.Sequential(
                    nn.Conv2d(
                        512, hidden_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(hidden_dim),
                ),
                "layer3": nn.Sequential(
                    nn.Conv2d(
                        1024, hidden_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(hidden_dim),
                ),
                "layer4": nn.Sequential(
                    nn.Conv2d(
                        2048, hidden_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(hidden_dim),
                ),
            }
        )

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            normalize_before=normalize_before,
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        self.conv_fuse = nn.Sequential(nn.Conv2d(
            hidden_dim * 2, hidden_dim, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(hidden_dim))

        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim*3,
                 atrous_rates=[6, 12, 24], out_channels=256),
            ConvUpsample(in_channels=256,
                         out_channels=1,
                         channels=[256, 256, 128],
                         upsample=[True, True, True],
                         drop=0.2,
                         ),
            nn.Upsample(
                size=(mask_dim, mask_dim), mode="bilinear", align_corners=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, frames, text, img_mask, text_mask):

        image = self.vision_encoder(frames)
        # print(image.shape)

        image_features = []
        for key in self.conv_3x3:
            layer_output = F.relu(self.conv_3x3[key](self.pool(image[key])))
            image_features.append(layer_output)

        B, C, H, W = image_features[-1].shape
        # img_mask = repeat(self.frame_mask, "b n -> (repeat b) n", repeat=B)

        f_text = self.text_encoder(text)
        f_text = f_text.permute(0, 2, 1)
        _, E, L = f_text.shape

        pos_embed_img = positionalencoding2d(B, d_model=C, height=H, width=W)
        pos_embed_img = pos_embed_img.flatten(2).permute(2, 0, 1)

        pos_embed_txt = positionalencoding1d(
            B, d_model=E, max_len=text_mask.shape[1])
        pos_embed_txt = pos_embed_txt.permute(1, 0, 2)

        pos_embed = torch.cat([pos_embed_img, pos_embed_txt], dim=0)

        joint_features = []
        for i in range(len(image_features)):
            f_img = image_features[i]
            B, C, H, W = f_img.shape

            f_img = f_img.flatten(2)

            f_joint = torch.cat([f_img, f_text], dim=2)
            src = f_joint.flatten(2).permute(2, 0, 1)

            src_key_padding_mask = ~torch.cat(
                [img_mask, text_mask], dim=1).bool()

            enc_out = self.transformer_encoder(
                src, pos=pos_embed, src_key_padding_mask=src_key_padding_mask
            )
            enc_out = enc_out.permute(1, 2, 0)

            f_img_out = enc_out[:, :, : H * W].view(B, C, H, W)

            f_txt_out = enc_out[:, :, H * W:].transpose(1, 2)  # B, L, E
            masked_sum = f_txt_out * text_mask[:, :, None]
            f_txt_out = masked_sum.sum(
                dim=1) / text_mask.sum(dim=-1, keepdim=True)

            f_out = torch.cat(
                [f_img_out, f_txt_out[:, :, None, None].expand(B, -1, H, W)], dim=1
            )

            enc_out = F.relu(self.conv_fuse(f_out))

            joint_features.append(enc_out)

        fused_feature = torch.cat(joint_features, dim=1)

        segm_mask = self.mm_decoder(fused_feature)  # .squeeze(1)

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
    def forward(self, frames, text, frame_mask, text_mask, position):

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
