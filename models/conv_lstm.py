import imp
from matplotlib.cm import ScalarMappable
import torch.nn as nn
import torch

from models.attentions import (
    CustomizingAttention,
    DotProductAttention,
    MultiHeadAttention,
    RelativeMultiHeadAttention,
    ScaledDotProductAttention,
)

from .mask_decoder import *
from .position_encoding import *
from einops import rearrange, repeat


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
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
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        mask_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=False,
        bias=True,
        return_all_layers=False,
        attn_type="dot_product",
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        self.hidden_feat = hidden_dim

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.attn_type = attn_type

        if self.attn_type == "dot_product":
            self.attention = DotProductAttention(self.hidden_feat)
        elif self.attn_type == "scaled_dot_product":
            self.attention = ScaledDotProductAttention(self.hidden_feat)
        elif self.attn_type == "multi_head":
            self.attention = MultiHeadAttention(d_model=self.hidden_feat, num_heads=8)
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
        
        
        # self.command_classifier = nn.Sequential(
        #     nn.Conv2d(3*self.hidden_feat, self.hidden_feat, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(self.hidden_feat, self.hidden_feat, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(self.hidden_feat, 1, kernel_size=1, stride=1, padding=0),
        #     # nn.Flatten(dim=1),
        #     # nn.Softmax(dim=1)
        #     nn.Sigmoid()
        # )
        
        self.lang_project = nn.Linear(768, self.hidden_dim)

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

        self.mask_decoder = nn.Sequential(
            ASPP(
                in_channels=self.hidden_dim[-1],
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
            nn.Upsample(size=(mask_dim, mask_dim), mode="bilinear", align_corners=True),
            nn.Sigmoid(),
        )

    def forward(
        self, anchors, positive_anchors, negative_anchors, frame_masks, positive_anchor_masks, negative_anchor_masks, hidden_state=None
    ):
        
        if not self.batch_first: 
            anchors = rearrange(anchors, "t b c h w -> b t c h w")

        b, _, c, h, w = anchors.shape
        l = positive_anchors.shape[2]

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        mask_list = []

        seq_len = anchors.size(1)
        cur_layer_input = anchors

        vis_pos_embd = positionalencoding2d(b, c, height=7, width=7)
        vis_pos_embd = rearrange(vis_pos_embd, "b c h w -> b (h w) c")

        for layer_idx in range(self.num_layers):

            hidden, cell = hidden_state[layer_idx]
            output_inner = []

            attn = None
            
            import pdb; pdb.set_trace()

            for t in range(seq_len):

                hidden, cell = self.cell_list[layer_idx](
                    input_tensor=anchors[:, t],
                    cur_state=[hidden, cell],
                )
                
                anchor = rearrange(hidden, "b c h w -> b (h w) c")
                
                positive_anchor = self.lang_project(positive_anchors[:, t])
                positive_anchor_mask = positive_anchor_masks[:, t]
                
                negative_anchor = self.lang_project(negative_anchors[:, t])
                
                if self.attn_type == "dot_product":
                    multi_modal_tensor, attn = self.attention(anchor, positive_anchor)
                elif self.attn_type == "scaled_dot_product":
                    multi_modal_tensor, attn = self.attention(anchor, positive_anchor, positive_anchor, padding)
                elif self.attn_type == "multi_head":
                    multi_modal_tensor, attn = self.attention(anchor, positive_anchor, positive_anchor, padding)
                # elif self.attn_type == "rel_multi_head":
                #     combined_tensor = torch.concat([hidden, positive_anchor], dim=1)
                #     # combined_pos_embd = torch.concat([vis_pos_embd, txt_pos_embd], dim=1)

                #     # multi_modal_tensor, attn = self.attention(combined_tensor, combined_tensor, combined_tensor, combined_pos_embd, combined_padding)
                #     multi_modal_tensor = multi_modal_tensor[:, :h*w]
                elif self.attn_type == "custom_attn":
                    # import pdb; pdb.set_trace()
                    # padding = repeat(padding, "b n l -> (rep b) n l", rep=8)
                    multi_modal_tensor, attn = self.attention(hidden, pos_anchor, attn, padding)
                else:
                    raise NotImplementedError(f'{self.attn_type} not implemented!')
                
                
                if layer_idx == self.num_layers - 1:
                    mask = self.mask_decoder(hidden)
                    mask_list.append(mask)
                
                hidden = rearrange(hidden, "b (h w) c -> b c h w", h=h, w=w)
                output_inner.append(hidden)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([hidden, cell])

        final_mask = torch.stack(mask_list, dim=2)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return last_state_list, final_mask

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == "__main__":
    convlstm = ConvLSTM(
        32,
        56,
        32,
        (3, 3),
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    )
    video = torch.rand(2, 5, 32, 14, 14)
    language = torch.rand(2, 10, 32)
    layer_out, last_state, frame_masks = convlstm(video, language)
    print(frame_masks.shape)
