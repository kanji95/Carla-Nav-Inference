import imp
from matplotlib.pyplot import text

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .transformer import *
from .position_encoding import *
from .mask_decoder import *
from .conv_lstm import ConvLSTM
from timesformer.models.vit import TimeSformer
from .clip4clip_modules.module_clip import CLIP


# simplest thing should be to predict a segmentation mask first
class SegmentationBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(
        self,
        vision_encoder,
        hidden_dim=384,
        image_dim=112,
        mask_dim=112,
        backbone="vit",
        imtext_matching="cross_attention",
    ):
        super(SegmentationBaseline, self).__init__()

        self.backbone = backbone

        self.imtext_matching = imtext_matching

        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        if self.imtext_matching == "concat":
            self.concat_decoder = nn.Sequential(
                nn.Linear(image_dim + hidden_dim, image_dim),
            )

        # self.mm_fusion = None

        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 6, 12, 24], out_channels=256),
            ConvUpsample(
                in_channels=256,
                out_channels=1,
                channels=[256, 256, 128],
                upsample=[True, True, True],
                drop=0.2,
            ),
            nn.Upsample(size=(mask_dim, mask_dim),
                        mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )

    def forward(self, frames, text, frame_mask, text_mask):

        vision_feat = self.vision_encoder(frames)
        if self.backbone.startswith("deeplabv3_"):
            vision_feat = rearrange(vision_feat, "b c h w -> b (h w) c")

        vision_feat = F.normalize(vision_feat, p=2, dim=1)  # B x N x C
        vision_dim = int(vision_feat.shape[1] ** 0.5)

        # vision_feat = rearrange(vision_feat, "b (h w) c -> b c h w", h=14, w=14)

        text_feat = self.text_encoder(text)  # B x L x C
        text_feat = F.normalize(text_feat, p=2, dim=1)  # B x L x C
        text_feat = text_feat * text_mask[:, :, None]

        # import pdb; pdb.set_trace()
        # print(vision_feat.shape, text_feat.shape)

        if self.imtext_matching == "cross_attention":
            cross_attn = torch.bmm(
                vision_feat, text_feat.transpose(1, 2).contiguous()
            )  # B x N x L
            cross_attn = cross_attn.softmax(dim=-1)
            attn_feat = cross_attn @ text_feat  # B x N x C

            fused_feat = vision_feat * attn_feat

        elif self.imtext_matching == "concat":
            concat = torch.concat(
                [vision_feat, text_feat], axis=1)  # B x L+N x C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        elif self.imtext_matching == "avg_concat":
            concat = torch.concat(
                [
                    vision_feat,
                    torch.mean(text_feat, dim=1).repeat(
                        1, vision_feat.shape[1], 1),
                ],
                axis=2,
            )  # B x N x 2C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        fused_feat = rearrange(
            fused_feat, "b (h w) c -> b c h w", h=vision_dim, w=vision_dim
        )

        segm_mask = self.mm_decoder(fused_feat)  # .squeeze(1)

        return segm_mask


# simplest thing should be to predict a segmentation mask first


class JointSegmentationBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(
        self,
        vision_encoder,
        hidden_dim=384,
        image_dim=112,
        mask_dim=112,
        traj_dim=56,
        backbone="vit",
        num_encoder_layers=2,
        normalize_before=True,
    ):
        super(JointSegmentationBaseline, self).__init__()

        self.backbone = backbone

        self.hidden_feat = hidden_dim
        # self.imtext_matching = imtext_matching

        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        encoder_layer = TransformerEncoderLayer(
            self.hidden_feat,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            normalize_before=normalize_before,
        )
        encoder_norm = nn.LayerNorm(
            self.hidden_feat) if normalize_before else None
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self.conv_fuse = nn.Conv2d(
            self.hidden_feat * 2,
            self.hidden_feat,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 6, 12, 24], out_channels=256),
            ConvUpsample(
                in_channels=256,
                out_channels=2,
                channels=[256, 256, 128],
                upsample=[True, True, True],
                drop=0.2,
            ),
            nn.Upsample(size=(mask_dim, mask_dim),
                        mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )

        self.traj_decoder = nn.Sequential(
            # ASPP(in_channels=hidden_dim, atrous_rates=[6, 12, 24], out_channels=256),
            ConvUpsample(
                in_channels=384,
                out_channels=1,
                channels=[256, 256, 128],
                upsample=[True, True, False],
                drop=0.2,
            ),
            nn.Upsample(size=(traj_dim, traj_dim),
                        mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )

    def forward(self, frames, text, frame_mask, text_mask):

        vision_feat = self.vision_encoder(frames)

        b, n, c = vision_feat.shape
        h = w = 14

        vision_feat = F.normalize(vision_feat, p=2, dim=1)

        text_feat = self.text_encoder(text)
        # text_feat = F.normalize(text_feat, p=2, dim=1)
        l = text_feat.shape[1]

        vis_pos_embd = positionalencoding2d(b, c, height=h, width=w)
        vis_pos_embd = rearrange(vis_pos_embd, "b c h w -> b (h w) c")

        txt_pos_embd = positionalencoding1d(b, c, max_len=l)

        combined_pos_embd = torch.cat([vis_pos_embd, txt_pos_embd], dim=1)
        combined_pos_embd = rearrange(combined_pos_embd, "b l c -> l b c")

        frame_tensor = rearrange(vision_feat, "b l c -> l b c")

        lang_tensor = rearrange(text_feat, "b l c -> l b c")

        combined_padding = ~torch.cat(
            [frame_mask, text_mask], dim=-1
        ).bool()

        combined_tensor = torch.cat([frame_tensor, lang_tensor], dim=0)
        enc_out = self.transformer_encoder(
            combined_tensor,
            pos=combined_pos_embd,
            src_key_padding_mask=combined_padding,
        )
        enc_out = enc_out.permute(1, 2, 0)

        f_img_out = enc_out[:, :, : h * w].view(b, c, h, w)

        f_txt_out = enc_out[:, :, h * w:].transpose(1, 2)  # B, L, E
        f_txt_out = f_txt_out.mean(dim=1)

        f_out = torch.cat(
            [f_img_out, f_txt_out[:, :, None, None].expand(b, -1, h, w)], dim=1
        )

        enc_out = F.relu(self.conv_fuse(f_out))

        segm_mask = self.mm_decoder(enc_out)
        traj_mask = self.traj_decoder(enc_out)

        return segm_mask, traj_mask


class VideoSegmentationBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(
        self,
        vision_encoder,
        hidden_dim=768,
        image_dim=112,
        mask_dim=112,
        spatial_dim=14,
        num_frames=16,
        imtext_matching="cross_attention",
    ):
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

        if self.imtext_matching == "concat":
            self.concat_decoder = nn.Sequential(
                nn.Linear(image_dim + hidden_dim, image_dim),
            )

        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 6, 12, 24], out_channels=256),
            ConvUpsample(
                in_channels=256,
                out_channels=1,
                channels=[256, 256, 128],
                upsample=[True, True, True],
                drop=0.2,
            ),
            nn.Upsample(size=(mask_dim, mask_dim),
                        mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )

    def forward(self, frames, text, frame_mask, text_mask):

        vision_feat, _ = self.vision_encoder(frames)  # B, N, C
        vision_feat = F.normalize(vision_feat, p=2, dim=1)  # B x N x C
        vision_feat = rearrange(
            vision_feat,
            "b (t h w) c -> (b t) (h w) c",
            t=self.num_frames,
            h=self.spatial_dim,
            w=self.spatial_dim,
        )

        text_feat = self.text_encoder(text)  # B x L x C
        text_feat = F.normalize(text_feat, p=2, dim=1)  # B x L x C
        text_feat = text_feat * text_mask[:, :, None]
        text_feat = repeat(text_feat, "b l c -> b t l c", t=self.num_frames)
        text_feat = rearrange(text_feat, "b t l c -> (b t) l c")

        if self.imtext_matching == "cross_attention":
            cross_attn = torch.bmm(
                vision_feat, text_feat.transpose(1, 2).contiguous()
            )  # B x N x L
            cross_attn = cross_attn.softmax(dim=-1)
            attn_feat = cross_attn @ text_feat  # B x N x C

            fused_feat = vision_feat * attn_feat

        elif self.imtext_matching == "concat":
            concat = torch.concat(
                [vision_feat, text_feat], axis=1)  # B x L+N x C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        elif self.imtext_matching == "avg_concat":
            concat = torch.concat(
                [
                    vision_feat,
                    torch.mean(text_feat, dim=1).repeat(
                        1, vision_feat.shape[1], 1),
                ],
                axis=2,
            )  # B x N x 2C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        fused_feat = rearrange(
            fused_feat,
            "(b t) (h w) c -> b c t h w",
            t=self.num_frames,
            h=self.spatial_dim,
            w=self.spatial_dim,
        )

        segm_mask = self.mm_decoder(fused_feat.mean(dim=2))  # .squeeze(1)

        return segm_mask


class JointVideoSegmentationBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(
        self,
        vision_encoder,
        hidden_dim=768,
        image_dim=112,
        mask_dim=112,
        traj_dim=56,
        spatial_dim=14,
        num_frames=16,
        imtext_matching="cross_attention",
    ):
        super(JointVideoSegmentationBaseline, self).__init__()

        self.spatial_dim = spatial_dim
        self.num_frames = num_frames

        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        self.imtext_matching = imtext_matching

        if self.imtext_matching == "concat":
            self.concat_decoder = nn.Sequential(
                nn.Linear(image_dim + hidden_dim, image_dim),
            )

        self.mm_decoder = nn.Sequential(
            # ASPP(in_channels=hidden_dim, atrous_rates=[6, 12, 24], out_channels=256),
            VideoUpsample(
                in_channels=hidden_dim,
                out_channels=2,
                channels=[256, 256, 128],
                upsample=[True, True, True],
                drop=0.2,
            ),
            nn.Upsample(
                size=(num_frames, mask_dim, mask_dim),
                mode="trilinear",
                align_corners=True,
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
            ConvUpsample(
                in_channels=256,
                out_channels=1,
                channels=[256, 256],
                upsample=[True, True],
                drop=0.2,
            ),
            nn.Upsample(size=(traj_dim, traj_dim),
                        mode="bilinear", align_corners=True),
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

        if self.imtext_matching == "cross_attention":
            cross_attn = torch.bmm(
                vision_feat, text_feat.transpose(1, 2).contiguous()
            )  # B x N x L
            cross_attn = cross_attn.softmax(dim=-1)
            attn_feat = cross_attn @ text_feat  # B x N x C

            fused_feat = vision_feat * attn_feat

        elif self.imtext_matching == "concat":
            concat = torch.concat(
                [vision_feat, text_feat], axis=1)  # B x L+N x C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        elif self.imtext_matching == "avg_concat":
            concat = torch.concat(
                [
                    vision_feat,
                    torch.mean(text_feat, dim=1).repeat(
                        1, vision_feat.shape[1], 1),
                ],
                axis=2,
            )  # B x N x 2C
            fused_feat = self.concat_decoder(concat)  # B x N x C

        # fused_feat = rearrange(fused_feat, "(b t) (h w) c -> b c t h w",
        #                        t=self.num_frames, h=self.spatial_dim, w=self.spatial_dim)
        fused_feat = rearrange(
            fused_feat,
            "b (t h w) c -> b c t h w",
            t=self.num_frames,
            h=self.spatial_dim,
            w=self.spatial_dim,
        )
        # fused_feat = fused_feat.mean(dim=2)

        segm_mask = self.mm_decoder(fused_feat)  # .squeeze(1)
        traj_mask = self.traj_decoder(fused_feat[:, :, -1])

        return segm_mask, traj_mask


class IROSBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(
        self,
        vision_encoder,
        hidden_dim=384,
        image_dim=112,
        mask_dim=112,
        num_encoder_layers=2,
        normalize_before=True,
        imtext_matching="cross_attention",
    ):
        super(IROSBaseline, self).__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        # self.frame_mask = torch.ones(1, 14*14, dtype=torch.int64)

        self.pool = nn.AdaptiveMaxPool2d((28, 28))
        self.conv_3x3 = nn.ModuleDict(
            {
                "layer2": nn.Sequential(
                    nn.Conv2d(512, hidden_dim, kernel_size=3,
                              stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                ),
                "layer3": nn.Sequential(
                    nn.Conv2d(1024, hidden_dim, kernel_size=3,
                              stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                ),
                "layer4": nn.Sequential(
                    nn.Conv2d(2048, hidden_dim, kernel_size=3,
                              stride=2, padding=1),
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
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
        )

        self.mm_decoder = nn.Sequential(
            ASPP(
                in_channels=hidden_dim * 3, atrous_rates=[6, 12, 24], out_channels=256
            ),
            ConvUpsample(
                in_channels=256,
                out_channels=2,
                channels=[256, 256, 128],
                upsample=[True, True, True],
                drop=0.2,
            ),
            nn.Upsample(size=(mask_dim, mask_dim),
                        mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )
        self.traj_decoder = nn.Sequential(
            ASPP(
                in_channels=hidden_dim * 3, atrous_rates=[6, 12, 24], out_channels=256
            ),
            ConvUpsample(
                in_channels=256,
                out_channels=1,
                channels=[256, 256, 128],
                upsample=[True, True, True],
                drop=0.2,
            ),
            nn.Upsample(size=(mask_dim, mask_dim),
                        mode="bilinear", align_corners=True),
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
        traj_mask = self.traj_decoder(fused_feature)  # .squeeze(1)

        return segm_mask, traj_mask


class ConvLSTMBaseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(
        self,
        vision_encoder,
        hidden_dim=768,
        image_dim=112,
        mask_dim=112,
        traj_dim=56,
        spatial_dim=14,
        num_frames=16,
        attn_type="dot_product",
    ):
        super(ConvLSTMBaseline, self).__init__()

        self.spatial_dim = spatial_dim
        self.num_frames = num_frames

        self.attn_type = attn_type

        self.vision_encoder = vision_encoder

        for param in self.vision_encoder.parameters():
            param.requires_grad_(False)

        # self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)
        self.sub_text_encoder = TextEncoder(
            num_layers=1, hidden_size=hidden_dim)

        self.conv3d = nn.Conv3d(
            192, hidden_dim, kernel_size=3, stride=1, padding=1)

        # self.bilinear = nn.Bilinear(
        #     self.num_frames * 49, 20, self.num_frames * 49)

        self.mm_decoder = ConvLSTM(
            input_dim=hidden_dim,
            mask_dim=mask_dim,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            return_all_layers=False,
            attn_type=self.attn_type,
        )

        # self.traj_decoder = nn.Sequential(
        #     ASPP(in_channels=hidden_dim, atrous_rates=[4, 6, 8], out_channels=256),
        #     ConvUpsample(
        #         in_channels=256,
        #         out_channels=1,
        #         channels=[256, 256],
        #         upsample=[True, True],
        #         drop=0.2,
        #     ),
        #     nn.Upsample(size=(traj_dim, traj_dim), mode="bilinear", align_corners=True),
        #     nn.Sigmoid(),
        # )

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
        sub_text_feat = rearrange(
            sub_text_feat, "(b n) l c -> b n l c", b=bs, n=nf)

        # import pdb; pdb.set_trace()
        last_state_feat, segm_mask = self.mm_decoder(
            vision_feat, sub_text_feat, frame_mask, sub_text_mask
        )  # .squeeze(1)

        # use last hidden state
        # traj_mask = self.traj_decoder(last_state_feat)

        return segm_mask, None


class CLIP_Baseline(nn.Module):
    def __init__(
        self,
        clip_dict,
        hidden_dim=512,
        image_dim=112,
        mask_dim=112,
        traj_dim=56,
        spatial_dim=14,
        num_frames=16,
        attn_type="dot_product",
        num_encoder_layers=2,
        normalize_before=True,
    ):
        super(CLIP_Baseline, self).__init__()

        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        self.num_frames = num_frames

        self.attn_type = attn_type

        self.clip_from_dict(clip_dict)

        encoder_layer = TransformerEncoderLayer(
            self.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            normalize_before=normalize_before,
        )
        encoder_norm = nn.LayerNorm(
            self.hidden_dim) if normalize_before else None
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self.conv_fuse = nn.Conv2d(
            self.hidden_dim * 2,
            self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(
                self.num_frames, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(),
            Rearrange('b c 1 h w -> b c h w'),
        )

        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 4, 6, 8], out_channels=256),
            ConvUpsample(
                in_channels=256,
                out_channels=2,
                channels=[256, 256, 128],
                upsample=[True, True, True],
                drop=0.2,
            ),
            nn.Upsample(size=(mask_dim, mask_dim),
                        mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )

        self.traj_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 4, 6, 8], out_channels=256),
            ConvUpsample(
                in_channels=256,
                out_channels=1,
                channels=[256, 256],
                upsample=[True, True],
                drop=0.2,
            ),
            nn.Upsample(size=(traj_dim, traj_dim),
                        mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )

    def forward(self, frames, text, frame_mask, text_mask):

        tok_type_ids = (text*0).detach().clone()
        # import pdb; pdb.set_trace()

        text_feat, vision_feat = self.get_sequence_visual_output(
            text, tok_type_ids, text_mask, frames, frame_mask)

        t = vision_feat.shape(1)

        text_feat = repeat(text_feat, "b c -> b t c", t=vision_feat.shape(1))

        text_feat = self.text_encoder(text)
        l = text_feat.shape[1]

        text_feat = repeat(text_feat, "b l c -> (b repeat) c l", repeat=t)

        vis_pos_embd = positionalencoding2d(b*t, c, height=h, width=w)
        vis_pos_embd = rearrange(vis_pos_embd, "b c h w -> b (h w) c")

        txt_pos_embd = positionalencoding1d(b*t, c, max_len=l)

        combined_pos_embd = torch.cat([vis_pos_embd, txt_pos_embd], dim=1)
        combined_pos_embd = rearrange(combined_pos_embd, "b l c -> l b c")

        frame_tensor = rearrange(vision_feat, "b c l -> l b c")

        lang_tensor = rearrange(text_feat, "b c l -> l b c")

        frame_mask = repeat(frame_mask, "b l -> (b repeat) l", repeat=t)
        text_mask = repeat(text_mask, "b l -> (b repeat) l", repeat=t)

        combined_padding = ~torch.cat(
            [frame_mask, text_mask], dim=-1
        ).bool()

        combined_tensor = torch.cat([frame_tensor, lang_tensor], dim=0)
        enc_out = self.transformer_encoder(
            combined_tensor,
            pos=combined_pos_embd,
            src_key_padding_mask=combined_padding,
        )
        enc_out = enc_out.permute(1, 2, 0)

        f_img_out = enc_out[:, :, : h * w].view(b*t, c, h, w)

        f_txt_out = enc_out[:, :, h * w:].transpose(1, 2)  # B, L, E
        f_txt_out = f_txt_out.mean(dim=1)

        f_out = torch.cat(
            [f_img_out, f_txt_out[:, :, None, None].expand(b*t, -1, h, w)], dim=1
        )

        enc_out = F.relu(self.conv_fuse(f_out))
        # enc_out = rearrange(enc_out, "(b t) c h w -> b c t h w", t=t)
        # enc_out = self.temporal_conv(enc_out)

        # import pdb; pdb.set_trace()
        segm_mask = self.mm_decoder(enc_out)

        # import pdb; pdb.set_trace()
        enc_out = rearrange(enc_out, "(b t) c h w -> b c t h w", t=t)
        traj_mask = self.traj_decoder(self.temporal_conv(enc_out))

        segm_mask = rearrange(segm_mask, "(b t) c h w -> b t c h w", t=t)
        # traj_mask = rearrange(traj_mask, "(b t) c h w -> b c t h w", t=t)

        return segm_mask, traj_mask

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(
            bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, channel, ts, h, w = video.shape
            video = video.view(b * ts, channel, h, w)
            video_frame = ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(
            video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, channel, ts, h, w = video.shape
            video = video.view(b * ts, channel, h, w)
            video_frame = ts

        sequence_output = self.get_sequence_output(
            input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(
            video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output

    def clip_from_dict(self, clip_state_dict):
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round(
                (clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round(
                (clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + \
                1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(
            ".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        self.linear_patch = '2d'  # set between 2d and 3d

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        self.convert_weights(self.clip)

    def convert_weights(self, model: nn.Module):
        """Convert applicable model parameters to fp16"""

        def _convert_weights_to_fp16(l):
            if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                l.weight.data = l.weight.data.half()
                if l.bias is not None:
                    l.bias.data = l.bias.data.half()

            if isinstance(l, nn.MultiheadAttention):
                for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.half()

            for name in ["text_projection", "proj"]:
                if hasattr(l, name):
                    attr = getattr(l, name)
                    if attr is not None:
                        attr.data = attr.data.half()

        model.apply(_convert_weights_to_fp16)


class Conv3D_Baseline(nn.Module):
    """Some Information about MyModule"""

    def __init__(
        self,
        vision_encoder,
        hidden_dim=768,
        image_dim=112,
        mask_dim=112,
        traj_dim=56,
        spatial_dim=14,
        num_frames=16,
        attn_type="dot_product",
        num_encoder_layers=2,
        normalize_before=True,
    ):
        super(Conv3D_Baseline, self).__init__()

        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        self.num_frames = num_frames

        self.attn_type = attn_type

        self.vision_encoder = vision_encoder

        for param in self.vision_encoder.parameters():
            param.requires_grad_(False)

        self.text_encoder = TextEncoder(num_layers=1, hidden_size=hidden_dim)

        self.conv3d = nn.Conv3d(
            192, hidden_dim, kernel_size=3, stride=1, padding=1)

        encoder_layer = TransformerEncoderLayer(
            self.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            normalize_before=normalize_before,
        )
        encoder_norm = nn.LayerNorm(
            self.hidden_dim) if normalize_before else None
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        self.conv_fuse = nn.Conv2d(
            self.hidden_dim * 2,
            self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(
                self.num_frames, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.ReLU(),
            Rearrange('b c 1 h w -> b c h w'),
        )

        self.mm_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 4, 6, 8], out_channels=256),
            ConvUpsample(
                in_channels=256,
                out_channels=2,
                channels=[256, 256, 128],
                upsample=[True, True, True],
                drop=0.2,
            ),
            nn.Upsample(size=(mask_dim, mask_dim),
                        mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )

        self.traj_decoder = nn.Sequential(
            ASPP(in_channels=hidden_dim, atrous_rates=[
                 4, 6, 8], out_channels=256),
            ConvUpsample(
                in_channels=256,
                out_channels=1,
                channels=[256, 256],
                upsample=[True, True],
                drop=0.2,
            ),
            nn.Upsample(size=(traj_dim, traj_dim),
                        mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )

    def forward(self, frames, text, frame_mask, text_mask):

        # bs = frames.shape[0]
        # nf = self.num_frames

        vision_feat = self.vision_encoder(frames)
        vision_feat = F.relu(self.conv3d(vision_feat))
        b, c, t, h, w = vision_feat.shape

        vision_feat = rearrange(vision_feat, "b c t h w -> (b t) c (h w)")

        text_feat = self.text_encoder(text)
        l = text_feat.shape[1]

        text_feat = repeat(text_feat, "b l c -> (b repeat) c l", repeat=t)

        vis_pos_embd = positionalencoding2d(b*t, c, height=h, width=w)
        vis_pos_embd = rearrange(vis_pos_embd, "b c h w -> b (h w) c")

        txt_pos_embd = positionalencoding1d(b*t, c, max_len=l)

        combined_pos_embd = torch.cat([vis_pos_embd, txt_pos_embd], dim=1)
        combined_pos_embd = rearrange(combined_pos_embd, "b l c -> l b c")

        frame_tensor = rearrange(vision_feat, "b c l -> l b c")

        lang_tensor = rearrange(text_feat, "b c l -> l b c")

        frame_mask = repeat(frame_mask, "b l -> (b repeat) l", repeat=t)
        text_mask = repeat(text_mask, "b l -> (b repeat) l", repeat=t)

        combined_padding = ~torch.cat(
            [frame_mask, text_mask], dim=-1
        ).bool()

        combined_tensor = torch.cat([frame_tensor, lang_tensor], dim=0)
        enc_out = self.transformer_encoder(
            combined_tensor,
            pos=combined_pos_embd,
            src_key_padding_mask=combined_padding,
        )
        enc_out = enc_out.permute(1, 2, 0)

        f_img_out = enc_out[:, :, : h * w].view(b*t, c, h, w)

        f_txt_out = enc_out[:, :, h * w:].transpose(1, 2)  # B, L, E
        f_txt_out = f_txt_out.mean(dim=1)

        f_out = torch.cat(
            [f_img_out, f_txt_out[:, :, None, None].expand(b*t, -1, h, w)], dim=1
        )

        enc_out = F.relu(self.conv_fuse(f_out))
        # enc_out = rearrange(enc_out, "(b t) c h w -> b c t h w", t=t)
        # enc_out = self.temporal_conv(enc_out)

        # import pdb; pdb.set_trace()
        segm_mask = self.mm_decoder(enc_out)

        # import pdb; pdb.set_trace()
        enc_out = rearrange(enc_out, "(b t) c h w -> b c t h w", t=t)
        traj_mask = self.traj_decoder(self.temporal_conv(enc_out))

        segm_mask = rearrange(segm_mask, "(b t) c h w -> b t c h w", t=t)
        # traj_mask = rearrange(traj_mask, "(b t) c h w -> b c t h w", t=t)

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
