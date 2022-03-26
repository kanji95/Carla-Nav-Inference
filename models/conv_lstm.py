import torch.nn as nn
import torch

from .attentions import *

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

        combined = torch.cat([input_tensor, h_cur], dim=1)
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
        
        self.lang_project = nn.Linear(768, self.hidden_dim[0])

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
                out_channels=1,
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
        
        # if not self.batch_first: 
        anchors = rearrange(anchors, "b c t h w -> b t c h w")

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

        txt_pos_embd = positionalencoding1d(b, c, max_len=l)

        for layer_idx in range(self.num_layers):

            hidden, cell = hidden_state[layer_idx]
            output_inner = []

            attn = None
            
            # import pdb; pdb.set_trace()
            
            anchor_tensors = []
            positive_anchor_tensors = []
            negative_anchor_tensors = []

            for t in range(seq_len):

                hidden, cell = self.cell_list[layer_idx](
                    input_tensor=anchors[:, t],
                    cur_state=[hidden, cell],
                )
                
                anchor = rearrange(hidden, "b c h w -> b (h w) c")
                # anchor_tensors.append(anchor)
                
                positive_anchor = self.lang_project(positive_anchors[:, t])
                # positive_anchor_tensors.append(positive_anchor)
                
                positive_anchor_mask = positive_anchor_masks[:, t]
                        
                negative_anchor = self.lang_project(negative_anchors[:, t])
                # negative_anchor_tensors.append(negative_anchor)
                
                negative_anchor_mask = negative_anchor_masks[:, t]
                
                anchor_feat = anchor.mean(dim=1)
                anchor_tensors.append(anchor_feat)
                
                pos_anchor_feat = positive_anchor.mean(dim=2).squeeze(1)
                positive_anchor_tensors.append(pos_anchor_feat)
                    
                neg_anchor_feat = negative_anchor.mean(dim=2).squeeze(0)
                # print(negative_anchor.mean(dim=2).shape)
                negative_anchor_tensors.append(neg_anchor_feat)
                
                # import pdb; pdb.set_trace()
                # print(anchor_feat.shape, pos_anchor_feat.shape, neg_anchor_feat.shape)
                score_ap = F.cosine_similarity(anchor_feat, pos_anchor_feat, dim=1)
                score_an = F.cosine_similarity(anchor_feat, neg_anchor_feat, dim=1)

                scores = torch.cat([score_ap, score_an])
                attn = F.softmax(scores, dim=0) 

                lang_tensor = torch.cat([positive_anchor, negative_anchor], dim=1) 
                wt_lang_tensor = (attn[None, :, None, None]*lang_tensor).sum(dim=1)
                
                if self.attn_type == "dot_product":
                    mm_tensor, attn = self.attention(anchor, wt_lang_tensor)
                elif self.attn_type == "scaled_dot_product":
                    mm_tensor, attn = self.attention(anchor, wt_lang_tensor, wt_lang_tensor)
                elif self.attn_type == "multi_head":
                    mm_tensor, attn = self.attention(anchor, anchor, wt_lang_tensor)
                elif self.attn_type == "rel_multi_head":
                    combined_tensor = torch.concat([anchor, wt_lang_tensor], dim=1)
                    combined_pos_embd = torch.concat([vis_pos_embd, txt_pos_embd], dim=1)
                    mm_tensor, attn = self.attention(combined_tensor, combined_tensor, combined_tensor, combined_pos_embd)
                elif self.attn_type == "custom_attn":
                    mm_tensor, attn = self.attention(anchor, wt_lang_tensor, attn,)
                else:
                    raise NotImplementedError(f'{self.attn_type} not implemented!')
                
                
                if layer_idx == self.num_layers - 1:
                    mask = self.mask_decoder(hidden)
                    mask_list.append(mask)
                
                hidden = rearrange(mm_tensor, "b (h w) c -> b c h w", h=h, w=w)
                output_inner.append(hidden)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([hidden, cell])
            
        anchor_tensors = torch.stack(anchor_tensors, dim=1)
        positive_tensors = torch.stack(positive_anchor_tensors, dim=1)
        negative_tensors = torch.stack(negative_anchor_tensors, dim=1)

        final_mask = torch.stack(mask_list, dim=1)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return hidden, final_mask, anchor_tensors, positive_tensors, negative_tensors

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
        192,
        56,
        192,
        (3, 3),
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    )
    convlstm.eval()
    
    anchors = torch.rand(1, 20, 192, 7, 7)
    positive_anchors = torch.rand(1, 20, 1, 15, 768)
    negative_anchors = torch.rand(1, 20, 1, 15, 768)
    frame_masks = torch.ones(1, 49)
    positive_anchor_masks = torch.randint(0, 2, (1, 15))
    negative_anchor_masks = torch.randint(0, 2, (1, 20, 15))
    
    last_state_list, final_mask = convlstm(anchors, positive_anchors, negative_anchors, frame_masks, positive_anchor_masks, negative_anchor_masks)
    print(frame_masks.shape)
