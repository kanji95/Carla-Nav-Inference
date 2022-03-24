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


# simplest thing should be to predict a segmentation mask first
class SegmentationBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, vision_encoder, hidden_dim=384, image_dim=112, mask_dim=112, backbone="vit", imtext_matching="cross_attention"):
        super(SegmentationBaseline, self).__init__()

        self.backbone = backbone

        self.imtext_matching = imtext_matching

        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        if self.imtext_matching == 'concat':
            self.concat_decoder = nn.Sequential(
                nn.Linear(image_dim+hidden_dim, image_dim),
            )

        # self.mm_fusion = None

        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 6, 12, 24], out_channels=256),
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

        if self.imtext_matching == 'cross_attention':
            cross_attn = torch.bmm(vision_feat, text_feat.transpose(
                1, 2).contiguous())  # B x N x L
            cross_attn = cross_attn.softmax(dim=-1)
            attn_feat = cross_attn @ text_feat  # B x N x C

            fused_feat = vision_feat * attn_feat

        elif self.imtext_matching == 'concat':
            concat = torch.concat(
                [vision_feat, text_feat], axis=1)  # B x L+N x C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        elif self.imtext_matching == 'avg_concat':
            concat = torch.concat(
                [vision_feat, torch.mean(text_feat, dim=1).repeat(1, vision_feat.shape[1], 1)], axis=2)  # B x N x 2C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        fused_feat = rearrange(
            fused_feat, "b (h w) c -> b c h w", h=vision_dim, w=vision_dim)

        segm_mask = self.mm_decoder(fused_feat)  # .squeeze(1)

        return segm_mask

# simplest thing should be to predict a segmentation mask first


class JointSegmentationBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, vision_encoder, hidden_dim=384, image_dim=112, mask_dim=112, traj_dim=56, backbone="vit", imtext_matching="cross_attention"):
        super(JointSegmentationBaseline, self).__init__()

        self.backbone = backbone

        self.imtext_matching = imtext_matching

        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        # self.mm_fusion = None

        if self.imtext_matching == 'concat':
            self.concat_decoder = nn.Sequential(
                nn.Linear(image_dim+hidden_dim, image_dim),
            )

        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 6, 12, 24], out_channels=256),
            ConvUpsample(in_channels=256,
                         out_channels=2,
                         channels=[256, 256, 128],
                         upsample=[True, True, True],
                         drop=0.2,
                         ),
            nn.Upsample(
                size=(mask_dim, mask_dim), mode="bilinear", align_corners=True
            ),
            nn.Sigmoid(),
        )

        self.traj_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 6, 12, 24], out_channels=256),
            ConvUpsample(in_channels=256,
                         out_channels=1,
                         channels=[256, 256],
                         upsample=[True, True],
                         drop=0.2,
                         ),
            nn.Upsample(
                size=(traj_dim, traj_dim), mode="bilinear", align_corners=True
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

        if self.imtext_matching == 'cross_attention':
            cross_attn = torch.bmm(vision_feat, text_feat.transpose(
                1, 2).contiguous())  # B x N x L
            cross_attn = cross_attn.softmax(dim=-1)
            attn_feat = cross_attn @ text_feat  # B x N x C

            fused_feat = vision_feat * attn_feat

        elif self.imtext_matching == 'concat':
            concat = torch.concat(
                [vision_feat, text_feat], axis=1)  # B x L+N x C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        elif self.imtext_matching == 'avg_concat':
            concat = torch.concat(
                [vision_feat, torch.mean(text_feat, dim=1).repeat(1, vision_feat.shape[1], 1)], axis=2)  # B x N x 2C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        fused_feat = rearrange(
            fused_feat, "b (h w) c -> b c h w", h=vision_dim, w=vision_dim)

        segm_mask = self.mm_decoder(fused_feat)  # .squeeze(1)
        traj_mask = self.traj_decoder(fused_feat)

        return segm_mask, traj_mask


class VideoSegmentationBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, vision_encoder, hidden_dim=768, image_dim=112, mask_dim=112, spatial_dim=14, num_frames=16, imtext_matching="cross_attention"):
        super(VideoSegmentationBaseline, self).__init__()

        self.spatial_dim = spatial_dim
        self.num_frames = num_frames

        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        self.imtext_matching = imtext_matching

        # self.mm_fusion = nn.Sequential(
        #     nn.Conv3d(hidden_dim*2, hidden_dim, kernel_size=1, stride=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool3d((1, None, None))
        # )

        if self.imtext_matching == 'concat':
            self.concat_decoder = nn.Sequential(
                nn.Linear(image_dim+hidden_dim, image_dim),
            )

        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 6, 12, 24], out_channels=256),
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

        vision_feat, _ = self.vision_encoder(frames)  # B, N, C
        vision_feat = F.normalize(vision_feat, p=2, dim=1)  # B x N x C
        vision_feat = rearrange(vision_feat, "b (t h w) c -> (b t) (h w) c",
                                t=self.num_frames, h=self.spatial_dim, w=self.spatial_dim)

        text_feat = self.text_encoder(text)  # B x L x C
        text_feat = F.normalize(text_feat, p=2, dim=1)  # B x L x C
        text_feat = text_feat * text_mask[:, :, None]
        text_feat = repeat(text_feat, "b l c -> b t l c", t=self.num_frames)
        text_feat = rearrange(text_feat, "b t l c -> (b t) l c")

        if self.imtext_matching == 'cross_attention':
            cross_attn = torch.bmm(vision_feat, text_feat.transpose(
                1, 2).contiguous())  # B x N x L
            cross_attn = cross_attn.softmax(dim=-1)
            attn_feat = cross_attn @ text_feat  # B x N x C

            fused_feat = vision_feat * attn_feat

        elif self.imtext_matching == 'concat':
            concat = torch.concat(
                [vision_feat, text_feat], axis=1)  # B x L+N x C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        elif self.imtext_matching == 'avg_concat':
            concat = torch.concat(
                [vision_feat, torch.mean(text_feat, dim=1).repeat(1, vision_feat.shape[1], 1)], axis=2)  # B x N x 2C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        fused_feat = rearrange(fused_feat, "(b t) (h w) c -> b c t h w",
                               t=self.num_frames, h=self.spatial_dim, w=self.spatial_dim)

        segm_mask = self.mm_decoder(fused_feat.mean(dim=2))  # .squeeze(1)

        return segm_mask


class JointVideoSegmentationBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, vision_encoder, hidden_dim=768, image_dim=112, mask_dim=112, traj_dim=56, spatial_dim=14, num_frames=16, imtext_matching='cross_attention'):
        super(JointVideoSegmentationBaseline, self).__init__()

        self.spatial_dim = spatial_dim
        self.num_frames = num_frames

        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        self.imtext_matching = imtext_matching

        if self.imtext_matching == 'concat':
            self.concat_decoder = nn.Sequential(
                nn.Linear(image_dim+hidden_dim, image_dim),
            )

        self.mm_decoder = nn.Sequential(
            # ASPP(in_channels=hidden_dim, atrous_rates=[6, 12, 24], out_channels=256),
            VideoUpsample(in_channels=hidden_dim,
                          out_channels=2,
                          channels=[256, 256, 128],
                          upsample=[True, True, True],
                          drop=0.2,
                          ),
            nn.Upsample(
                size=(num_frames, mask_dim, mask_dim), mode="trilinear", align_corners=True
            ),
            nn.Sigmoid(),
        )
        
        # self.mm_decoder = ConvLSTM(
        #     input_dim=hidden_dim,
        #     mask_dim=mask_dim,
        #     hidden_dim=hidden_dim,
        #     kernel_size=3,
        #     num_layers=1,
        #     batch_first=True,
        # )

        self.traj_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 6, 12, 24], out_channels=256),
            ConvUpsample(in_channels=256,
                         out_channels=1,
                         channels=[256, 256],
                         upsample=[True, True],
                         drop=0.2,
                         ),
            nn.Upsample(
                size=(traj_dim, traj_dim), mode="bilinear", align_corners=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, frames, text, frame_mask, text_mask):

        vision_feat, _ = self.vision_encoder(frames)  # B, N, C
        vision_feat = F.normalize(vision_feat, p=2, dim=1)  # B x N x C
        # vision_feat = rearrange(vision_feat, "b (t h w) c -> (b t) (h w) c",
        #                         t=self.num_frames, h=self.spatial_dim, w=self.spatial_dim)

        text_feat = self.text_encoder(text)  # B x L x C
        text_feat = F.normalize(text_feat, p=2, dim=1)  # B x L x C
        text_feat = text_feat * text_mask[:, :, None]
        # text_feat = repeat(text_feat, "b l c -> b t l c", t=self.num_frames)
        # text_feat = rearrange(text_feat, "b t l c -> (b t) l c")

        if self.imtext_matching == 'cross_attention':
            cross_attn = torch.bmm(vision_feat, text_feat.transpose(
                1, 2).contiguous())  # B x N x L
            cross_attn = cross_attn.softmax(dim=-1)
            attn_feat = cross_attn @ text_feat  # B x N x C

            fused_feat = vision_feat * attn_feat

        elif self.imtext_matching == 'concat':
            concat = torch.concat(
                [vision_feat, text_feat], axis=1)  # B x L+N x C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        elif self.imtext_matching == 'avg_concat':
            concat = torch.concat(
                [vision_feat, torch.mean(text_feat, dim=1).repeat(1, vision_feat.shape[1], 1)], axis=2)  # B x N x 2C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        # fused_feat = rearrange(fused_feat, "(b t) (h w) c -> b c t h w",
        #                        t=self.num_frames, h=self.spatial_dim, w=self.spatial_dim)
        fused_feat = rearrange(fused_feat, "b (t h w) c -> b c t h w",
                               t=self.num_frames, h=self.spatial_dim, w=self.spatial_dim)
        # fused_feat = fused_feat.mean(dim=2)

        segm_mask = self.mm_decoder(fused_feat)  # .squeeze(1)
        traj_mask = self.traj_decoder(fused_feat[:, :, -1])

        return segm_mask, traj_mask


class IROSBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, vision_encoder, hidden_dim=384, image_dim=112, mask_dim=112, num_encoder_layers=2, normalize_before=True, imtext_matching='cross_attention'):
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


class ConvLSTMBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, vision_encoder, hidden_dim=768, image_dim=112, mask_dim=112, traj_dim=56, spatial_dim=14, num_frames=16, attn_type="dot_product"):
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
        # vision_feat = rearrange(vision_feat, "b c t h w -> b c (t h w)")

        # text_feat = self.text_encoder(text)
        # text_feat = rearrange(text_feat, "b l c -> b c l")
        
        # mm_feat = self.smm_feat, "b c (t h w) -> b t c h w", t=nf, h=7, w=7)
        
        sub_text = rearrange(sub_text, "b n l c -> (b n) l c")
        sub_text_feat = self.sub_text_encoder(sub_text)
        sub_text_feat = rearrange(sub_text_feat, "(b n) l c -> b n l c", b=bs, n=nf)

        # import pdb; pdb.set_trace()
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
