import imp
from matplotlib.cm import ScalarMappable
import torch.nn as nn
import torch

import math

from models.attentions import (
    CustomizingAttention,
    DotProductAttention,
    MultiHeadAttention,
    RelativeMultiHeadAttention,
    ScaledDotProductAttention,
)

from models.mask_decoder import *
from models.position_encoding import *
from einops import rearrange, repeat


class ConvAttn(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_frame: Number of frames recieved
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convattn = ConvAttn(64, 16, 3, 1, True, True, False)
        >> _, last_states = convattn(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        mask_dim,
        hidden_dim,
        num_frames,
        batch_first=False,
        bias=True,
        return_all_layers=False,
        attn_type="dot_product",
    ):
        super(ConvAttn, self).__init__()

        self.hidden_feat = hidden_dim

        self.input_dim = input_dim
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.attn_type = attn_type

        if self.attn_type == "dot_product":
            self.attention = DotProductAttention(self.hidden_feat)
        elif self.attn_type == "scaled_dot_product":
            self.attention = ScaledDotProductAttention(self.hidden_feat)
        elif self.attn_type == "multi_head":
            self.attention = MultiHeadAttention(
                d_model=self.hidden_feat, num_heads=8)
        elif self.attn_type == "rel_multi_head":
            self.attention = RelativeMultiHeadAttention(
                d_model=self.hidden_feat, num_heads=8
            )
        elif self.attn_type == "custom_attn":
            self.attention = CustomizingAttention(
                hidden_dim=self.hidden_feat,
                num_heads=8,
                conv_out_channel=self.hidden_feat,
            )
        else:
            raise NotImplementedError(f"{self.attn_type} not implemented!")

        if self.attn_type != 'dot_product':
            raise NotImplementedError(f"{self.attn_type} not implemented!")

        curr_frames = self.num_frames
        cell_list = []
        in_channels = self.hidden_dim
        kernel_size = (3, 5, 5)
        padding = (kernel_size[0]//2, kernel_size[1]//2, kernel_size[2]//2)
        stride = (2, 1, 1)

        while curr_frames > 1:
            cell_list.append(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=in_channels*2,
                          kernel_size=kernel_size,
                          padding=padding,
                          stride=stride
                          )
            )
            in_channels = 2*in_channels
            curr_frames = math.floor(
                (curr_frames+2*padding[0]-kernel_size[0])/stride[0]+1)
        cell_list.append(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=self.hidden_dim,
                      kernel_size=(1, *kernel_size[1:]),
                      padding=(0, *padding[1:])
                      )
        )
        self.cell_list = nn.Sequential(*cell_list,)

        self.mask_decoder = nn.Sequential(
            ASPP(
                in_channels=self.hidden_dim,
                atrous_rates=[4, 6, 8],
                out_channels=256,
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

    def forward(
            self, input_tensor, context_tensor):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        context_tensor: todo
            4-D Tensor of shape (b, n, l, c)
            b: batch size
            n: num_frames equivalent, n == t
            l: language command size
            c: word embedding size

        Returns
        -------
        frame
        """

        # import pdb; pdb.set_trace()

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = rearrange(input_tensor, "t b c h w -> b t c h w")

        b, t, c, h, w = input_tensor.shape
        _, n, l, _ = context_tensor.shape

        assert b == context_tensor.shape[0]
        assert n == t
        assert c == context_tensor.shape[3]

        #import pdb; pdb.set_trace()

        visual_tensor = rearrange(
            input_tensor, "b t c h w -> (b t) (h w) c")
        #import pdb; pdb.set_trace()

        # lang_tensor = repeat(context_tensor, 'b l c -> (b t) l c', t=t)
        lang_tensor = rearrange(context_tensor, 'b t l c -> (b t) l c', t=t)

        # import pdb; pdb.set_trace()
        multi_modal_tensor, attn = self.attention(
            visual_tensor, lang_tensor)

        multi_modal_tensor = rearrange(
            multi_modal_tensor, "(b t) (h w) c -> b c t h w", b=b, t=t, h=h, w=w)

        out_cnn = self.cell_list(multi_modal_tensor)

        out_cnn = out_cnn.squeeze(2)

        segm_mask = self.mask_decoder(out_cnn)

        # segm_mask = repeat(
        #     segm_mask, 'b c h w -> b c t h w', t=self.num_frames)

        if not self.return_all_layers:
            return segm_mask
        else:
            return segm_mask, out_cnn


if __name__ == "__main__":
    b = 2
    n = t = 32
    l = 10
    hidden_dim = 32
    h, w = 14, 14

    c = hidden_dim

    convattn = ConvAttn(
        h,
        56,
        hidden_dim=hidden_dim,
        num_frames=t,
        batch_first=True,
        bias=True,
        return_all_layers=True
    )
    video = torch.rand(b, t, c, h, w)
    language = torch.rand(b, n, l, c)
    frame_mask, out_cnn = convattn(video, language)
    print(frame_mask.shape, out_cnn.shape)
