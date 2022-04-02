import wandb

import os
import math
import random
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image

from tqdm import tqdm

import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from einops import rearrange

def print_(statement, default_gpu=True):
    if default_gpu:
        print(statement, flush=True)

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.5 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@torch.no_grad()
def grad_check(named_parameters):
    thresh = 0.001

    layers = []
    max_grads = []
    mean_grads = []
    max_colors = []
    mean_colors = []

    for n, p in named_parameters:
        # import pdb; pdb.set_trace()
        # print(n)
        if p.requires_grad and "bias" not in n:
            max_grad = p.grad.cpu().abs().max()
            mean_grad = p.grad.cpu().abs().mean()
            layers.append(".".join(n.split(".")[2:]))
            max_grads.append(max_grad)
            mean_grads.append(mean_grad)

    for i, (val_mx, val_mn) in enumerate(zip(max_grads, mean_grads)):
        if val_mx > thresh:
            max_colors.append("r")
        else:
            max_colors.append("g")
        if val_mn > thresh:
            mean_colors.append("b")
        else:
            mean_colors.append("y")
    ax = plt.subplot(111)
    x = np.arange(len(layers))
    w = 0.3

    ax.bar(x - w, max_grads, width=w, color=max_colors, align="center", hatch="////")
    ax.bar(x, mean_grads, width=w, color=mean_colors, align="center", hatch="----")

    plt.xticks(x - w / 2, layers, fontsize=5, rotation="vertical")
    plt.xlim(left=-1, right=len(layers))

    plt.ylim(bottom=0.0, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient Values")
    plt.title("Model Gradients")

    hatch_dict = {0: "////", 1: "----"}
    legends = []
    for i in range(len(hatch_dict)):
        p = patches.Patch(facecolor="#DCDCDC", hatch=hatch_dict[i])
        legends.append(p)

    ax.legend(legends, ["Max", "Mean"])

    plt.grid(True)
    plt.tight_layout()
    wandb.log({"Gradients": wandb.Image(plt)})
    plt.close()

@torch.no_grad()
def log_frame_predicitons(front_cam_image, lang_command, pred_mask, traj_mask, gt_mask, gt_traj_mask, episode_num, sample_idx, title="train", k=4
):
    indices = np.random.choice(range(pred_mask.shape[0]), size=k, replace=False)

    _, c, h, w = pred_mask.shape

    # import pdb; pdb.set_trace()

    figure, axes = plt.subplots(nrows=k, ncols=5)
    for i, index in enumerate(indices):
        index = indices[i]
        
        orig_img = front_cam_image[index]
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(lang_command[index], fontsize=5)
        axes[i, 0].set_axis_off()

        mask_pred = rearrange(pred_mask[index], "c h w -> h w c")

        mask_pred_ = np.zeros((h, w, 3))
        mask_pred_[:, :, 0] = mask_pred[:, :, 0] * 255  # red
        mask_pred_[:, :, 2] = mask_pred[:, :, 1] * 255  # blue
        mask_pred = np.uint8(mask_pred_)
        # mask_pred = np.uint8(mask_pred * 255)

        axes[i, 1].imshow(mask_pred)
        axes[i, 1].set_title(f"episode_{episode_num[index]}_idx_{sample_idx[index]}", fontsize=5)
        axes[i, 1].set_axis_off()
        
        mask_gt = rearrange(gt_mask[index], "c h w -> h w c")

        mask_gt_ = np.zeros((h, w, 3))
        if mask_gt[:, :, 0].max() > mask_gt[:, :, 1].max():
            mask_gt_[:, :, 1] = mask_gt[:, :, 0] * 255  # lime
        else:
            mask_gt_[:, :, 1] = mask_gt[:, :, 1] * 100  # dark green

        # mask_gt_[:, :, 0] = mask_gt[:, :, 0] * 100  # dark red
        # mask_gt_[:, :, 1] = mask_gt[:, :, 1] * 100  # dark green
        mask_gt = np.uint8(mask_gt_)
        
        # mask_gt = np.uint8(mask_gt * 255)

        axes[i, 2].imshow(mask_gt)
        axes[i, 2].set_title("GT Mask", fontsize=5)
        axes[i, 2].set_axis_off()

        traj_pred = rearrange(traj_mask[index], "c h w -> h w c")
        traj_pred = np.uint8(traj_pred * 255)
        axes[i, 3].imshow(traj_pred)
        axes[i, 3].set_title(f"Predicted Trajectory", fontsize=5)
        axes[i, 3].set_axis_off()

        traj_mask_gt = rearrange(gt_traj_mask[index], "c h w -> h w c")
        traj_mask_gt = np.uint8(traj_mask_gt * 255)
        axes[i, 4].imshow(traj_mask_gt)
        axes[i, 4].set_title("GT Traj Mask", fontsize=5)
        axes[i, 4].set_axis_off()

    figure.tight_layout()
    wandb.log({f"{title}_segmentation": wandb.Image(figure)}, commit=True)
    plt.close(figure)

# @torch.no_grad()
# def log_video_predicitons(front_cam_video, lang_command, pred_mask, traj_mask, gt_mask, gt_traj_mask, episode_num, sample_idx, title="train", k=4
# ):
#     indices = np.random.choice(range(pred_mask.shape[0]), size=k, replace=False)
# 
#     b, c, t, h, w = pred_mask.shape
#     # b, c, h, w = pred_mask.shape
#     
#     # figure, axes = plt.subplots(nrows=k, ncols=5)
#     for i, index in enumerate(indices):
#         index = indices[i]
#         
#         command = lang_command[index]
# 
#         orig_video = front_cam_video[index]
#         orig_video = rearrange(orig_video, "t h w c -> t c h w")
# 
#         mask_pred = rearrange(pred_mask[index], "c t h w -> t c h w")
#         mask_pred_ = np.zeros((t, 3, h, w))
#         mask_pred_[:, 0] = mask_pred[:, 0] * 255  # red
#         mask_pred_[:, 2] = mask_pred[:, 1] * 255  # blue
#         mask_pred = np.uint8(mask_pred_)
# 
#         # mask_pred = rearrange(pred_mask[index], "c h w -> h w c")
#         # mask_pred_ = np.zeros((h, w, 3))
#         # mask_pred_[:, :, 0] = mask_pred[:, :, 0] * 255  # red
#         # mask_pred_[:, :, 2] = mask_pred[:, :, 1] * 255  # blue
#         # mask_pred = np.uint8(mask_pred_)
#         
#         mask_gt = rearrange(gt_mask[index], "c t h w -> t c h w")
#         mask_gt_ = np.zeros((t, 3, h, w))
#         if mask_gt[:, 0].max() > mask_gt[:, 1].max():
#             mask_gt_[:, 1] = mask_gt[:, 0] * 100  # dark green
#         else:
#             mask_gt_[:, 1] = mask_gt[:, 1] * 255  # lime
#         mask_gt = np.uint8(mask_gt_)
# 
#         # mask_gt = rearrange(gt_mask[index], "c h w -> h w c")
#         # mask_gt_ = np.zeros((h, w, 3))
#         # if mask_gt[:, :, 0].max() > mask_gt[:, :, 1].max():
#         #     mask_gt_[:, :, 1] = mask_gt[:, :, 0] * 100  # dark green
#         # else:
#         #     mask_gt_[:, :, 1] = mask_gt[:, :, 1] * 255  # lime
#         # mask_gt = np.uint8(mask_gt_)
# 
#         traj_pred = rearrange(traj_mask[index], "c h w -> h w c")
#         traj_pred = np.uint8(traj_pred * 255)
# 
#         traj_mask_gt = rearrange(gt_traj_mask[index], "c h w -> h w c")
#         traj_mask_gt = np.uint8(traj_mask_gt * 255)
# 
#         wandb.log(
#             {
#                 f"{title}_video": wandb.Video(orig_video, fps=4, caption=command, format="mp4"),
#                 f"{title}_pred_mask": wandb.Video(
#                     mask_pred, fps=4, caption=command, format="mp4"
#                 ),
#                 f"{title}_gt_mask": wandb.Video(
#                     mask_gt, fps=4, caption=command, format="mp4"
#                 ),
#                 # f"{title}_pred_mask": wandb.Image(
#                 #     mask_pred, caption=command,
#                 # ),
#                 # f"{title}_gt_mask": wandb.Image(
#                 #     mask_gt, caption=command,
#                 # ),
#                 f"{title}_traj_mask": wandb.Image(
#                     traj_pred, caption=command
#                 ),
#                 f"{title}_traj_mask_gt": wandb.Image(
#                     traj_mask_gt, caption=command
#                 ),
#             }
#         )
        

@torch.no_grad()
def log_video_predicitons(front_cam_video, lang_command, pred_mask, traj_mask, gt_mask, gt_traj_mask, episode_num, sample_idx, title="train", k=4
):

    # import pdb; pdb.set_trace()
    # print(gt_mask.shape, pred_mask.shape)
    index = np.random.choice(pred_mask.shape[0])

    b, t, c, h, w = pred_mask.shape

    orig_video = rearrange(front_cam_video[index], "t h w c -> t c h w")

    mask_pred = np.zeros((t, 3, h, w))
    mask_pred[:, 0] = pred_mask[index, :, 0]
    mask_pred = np.uint8(mask_pred * 255)
    
    mask_gt = np.zeros((t, 3, h, w))
    mask_gt[:, 1] = gt_mask[index, :, 0]
    mask_gt = np.uint8(mask_gt * 255)

    # mask_pred = np.uint8(rearrange(gt_mask, "b t c h w -> t (b c) h w")*255.)    
    # mask_gt = np.uint8(rearrange(gt_mask[0], "t c w h -> t  c w h")*255.)

    # traj_pred = np.uint8(rearrange(traj_mask[index], "c h w -> h w c")*255.)
    traj_mask_gt = np.uint8(rearrange(gt_traj_mask[index], "c h w -> h w c")*255.)
    command = lang_command[0]

    wandb.log(
        {
            f"{title}_video": wandb.Video(orig_video, fps=4, caption=command, format="mp4"),
            f"{title}_pred_mask": wandb.Video(
                mask_pred, fps=4, caption=command, format="mp4"
            ),
            f"{title}_gt_mask": wandb.Video(
                mask_gt, fps=4, caption=command, format="mp4"
            ),
            # f"{title}_traj_mask": wandb.Image(
            #     traj_pred, caption=command
            # ),
            f"{title}_traj_mask_gt": wandb.Image(
                traj_mask_gt, caption=command
            ),
        }
    )
