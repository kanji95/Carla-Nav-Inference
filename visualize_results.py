import wandb

import re
import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
import timm
from timesformer.models.vit import VisionTransformer
import torchvision.transforms as transforms
from torchvision.models._utils import IntermediateLayerGetter

from models.model import *
from dataloader.word_utils import Corpus


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpu = torch.cuda.device_count()
    print(f"Using {device} with {num_gpu} GPUS!")

    experiment = wandb.init(project="Language Navigation", dir="/tmp")

    val_path = os.path.join(args.data_root, "val/")
    glove_path = args.glove_path
    checkpoint_path = args.checkpoint

    corpus = Corpus(glove_path)

    return_layers = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}

    mode = "image"
    if "vit_" in args.img_backbone:
        img_backbone = timm.create_model(args.img_backbone, pretrained=True)
        visual_encoder = nn.Sequential(*list(img_backbone.children())[:-1])
        network = JointSegmentationBaseline(
            visual_encoder,
            hidden_dim=args.hidden_dim,
            mask_dim=args.mask_dim,
            traj_dim=args.traj_dim,
            backbone=args.img_backbone,
        )
    elif "dino_resnet50" in args.img_backbone:
        img_backbone = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        visual_encoder = IntermediateLayerGetter(img_backbone, return_layers)
        network = IROSBaseline(
            visual_encoder, hidden_dim=args.hidden_dim, mask_dim=args.mask_dim
        )
    elif "timesformer" in args.img_backbone:
        mode = "video"
        spatial_dim = args.image_dim // args.patch_size
        visual_encoder = VisionTransformer(
            img_size=args.image_dim,
            patch_size=args.patch_size,
            embed_dim=args.hidden_dim,
            depth=2,
            num_heads=8,
            num_frames=args.num_frames,
        )
        network = JointVideoSegmentationBaseline(
            visual_encoder,
            hidden_dim=args.hidden_dim,
            mask_dim=args.mask_dim,
            traj_dim=args.traj_dim,
            spatial_dim=spatial_dim,
            num_frames=args.num_frames,
        )
    elif "deeplabv3_" in args.img_backbone:
        img_backbone = torch.hub.load(
            "pytorch/vision:v0.10.0", args.img_backbone, pretrained=True
        )
        visual_encoder = nn.Sequential(
            *list(img_backbone._modules["backbone"].children())
        )
        network = JointSegmentationBaseline(
            visual_encoder,
            hidden_dim=args.hidden_dim,
            mask_dim=args.mask_dim,
            traj_dim=args.traj_dim,
            backbone=args.img_backbone,
        )
    elif "convlstm" in args.img_backbone:
            mode = "video"
            spatial_dim = args.image_dim//args.patch_size
            # visual_encoder = VisionTransformer(img_size=args.image_dim, patch_size=args.patch_size,
            #                                   embed_dim=args.hidden_dim, depth=2, num_heads=8, num_frames=args.num_frames)
            video_encoder = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
            visual_encoder = nn.Sequential(*list(video_encoder.blocks.children())[:-1])

            network = ConvLSTMBaseline(
                visual_encoder,
                hidden_dim=args.hidden_dim,
                image_dim=args.image_dim,
                mask_dim=args.mask_dim,
                traj_dim=args.traj_dim,
                spatial_dim=spatial_dim,
                num_frames=args.num_frames,
                attn_type=args.attn_type,
            )
    wandb.watch(network, log="all")

    if num_gpu > 1:
        network = nn.DataParallel(network)
        print("Using DataParallel mode!")
    network.to(device)

    checkpoint = torch.load(checkpoint_path)
    network.load_state_dict(checkpoint["state_dict"])

    network.eval()

    img_transform = transforms.Compose(
        [
            transforms.Resize((args.image_dim, args.image_dim)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            #                      0.229, 0.224, 0.225]),
        ]
    )

    mask_transform = transforms.Compose(
        [
            transforms.Resize((args.mask_dim, args.mask_dim)),
            transforms.ToTensor(),
        ]
    )

    # traj_transform = transforms.Compose(
    #     [
    #         transforms.Resize((args.traj_dim, args.traj_dim)),
    #         transforms.ToTensor(),
    #     ]
    # )

    frame_mask = torch.ones(1, 7 * 7, dtype=torch.int64).cuda(non_blocking=True)

    # columns = ["Command"]

    episodes = glob(val_path + "*")
    episode_num = 0
    for episode in episodes:

        # command_path = episode + "/command.txt"
        # command = open(command_path, "r").read()
        # command = re.sub(r"[^\w\s]", "", command)
        
        episode_num = int(episode.split("/")[-1])

        command_df = pd.read_csv("./dataloader/sub_commands_val.csv", index_col=0)
        command = command_df.loc[episode_num]['command']
        command = re.sub(r"[^\w\s]", "", command)

        sub_commands = [command_df.loc[episode_num]['sub_command_0'], command_df.loc[episode_num]['sub_command_1']]
        if pd.isna(command_df.loc[episode_num]['sub_command_1']):
            sub_commands[1] = command_df.loc[episode_num]['sub_command_0']
        sub_commands = [re.sub(r"[^\w\s]", "", sub_command) for sub_command in sub_commands]

        phrase, phrase_mask = corpus.tokenize(command)
        phrase = phrase.unsqueeze(0)
        phrase_mask = phrase_mask.unsqueeze(0)

        sub_phrase = torch.stack([phrase]*args.num_frames, dim=1)
        sub_phrase_mask = torch.stack([phrase_mask]*args.num_frames, dim=1)

        # sub_tokens = []
        # sub_phrase_masks = []
        # for sub_command in sub_commands:
        #     sub_command = re.sub(r"[^\w\s]", "", sub_command.lower())
        #     sub_token, sub_phrase_mask = self.corpus.tokenize(sub_command)
        #     sub_tokens.append(sub_token)
        #     sub_phrase_masks.append(sub_phrase_mask)

        # sub_tokens = torch.stack(sub_tokens, dim=0)[None]
        # sub_phrase_masks = torch.stack(sub_phrase_masks, dim=0)[None]

        image_files = sorted(glob(episode + "/images/*.png"))
        mask_files = sorted(glob(episode + "/masks/*.png"))

        frame_video = []
        mask_video = []
        traj_video = []
        gt_mask_video = []

        if mode == "image":
            run_image_model(
                network,
                img_transform,
                mask_transform,
                frame_mask,
                phrase,
                sub_phrase,
                phrase_mask,
                sub_phrase_mask,
                image_files,
                mask_files,
                frame_video,
                mask_video,
                traj_video,
                gt_mask_video,
            )
        else:
            run_video_model(
                network,
                img_transform,
                mask_transform,
                frame_mask,
                phrase,
                sub_phrase,
                phrase_mask,
                sub_phrase_mask,
                image_files,
                mask_files,
                frame_video,
                mask_video,
                traj_video,
                gt_mask_video,
                num_frames = args.num_frames,
            )

        frame_video = np.concatenate(frame_video, axis=0)
        mask_video = np.concatenate(mask_video, axis=0)
        traj_video = np.concatenate(traj_video, axis=0)
        gt_mask_video = np.concatenate(gt_mask_video, axis=0)

        # import pdb; pdb.set_trace()
        mask_video_overlay = np.copy(frame_video)
        # print(mask_video_overlay.shape, mask_video.shape)
        mask_video_overlay[:, 0] += mask_video[:, 0] / mask_video[:, 0].max() # red - intermediate point
        # mask_video_overlay[:, 0] += mask_video[:, 0] / mask_video[:, 0].max() # red - intermediate point
        mask_video_overlay[:, 2] += mask_video[:, 1] / mask_video[:, 1].max() # blue - final point
        mask_video_overlay = np.clip(mask_video_overlay, a_min=0.0, a_max=1.0)

        traj_video_overlay = np.copy(frame_video)
        traj_video_overlay[:, 0] += traj_video[:, 0] / traj_video.max()
        traj_video_overlay = np.clip(traj_video_overlay, a_min=0.0, a_max=1.0)

        gt_mask_video_overlay = np.copy(frame_video)
        gt_mask_video_overlay[:, 0] += gt_mask_video[:, 0]
        gt_mask_video_overlay = np.clip(gt_mask_video_overlay, a_min=0.0, a_max=1.0)

        frame_video = np.uint8(frame_video * 255)
        mask_video = np.uint8(mask_video_overlay * 255)
        traj_video = np.uint8(traj_video_overlay * 255)
        gt_mask_video = np.uint8(gt_mask_video_overlay * 255)

        print(episode_num, command)

        wandb.log(
            {
                "video": wandb.Video(frame_video, fps=4, caption=command, format="mp4"),
                "pred_mask": wandb.Video(
                    mask_video, fps=4, caption=command, format="mp4"
                ),
                "traj_mask": wandb.Video(
                    traj_video, fps=4, caption=command, format="mp4"
                ),
                "gt_mask": wandb.Video(
                    gt_mask_video, fps=4, caption=command, format="mp4"
                ),
            }
        )

        episode_num += 1


def run_image_model(
    network,
    img_transform,
    mask_transform,
    frame_mask,
    phrase,
    sub_phrase,
    phrase_mask,
    sub_phrase_mask,
    image_files,
    mask_files,
    frame_video,
    mask_video,
    traj_video,
    gt_mask_video,
):
    for image_file, mask_file in zip(image_files, mask_files):
        image = Image.open(image_file).convert("RGB")
        gt_mask = Image.open(mask_file).convert("L")

        frame = img_transform(image).cuda(non_blocking=True).unsqueeze(0)
        gt_mask = mask_transform(gt_mask).unsqueeze(0) #.cuda(non_blocking=True).unsqueeze(0)

        mask, traj_mask = network(frame, phrase, frame_mask, phrase_mask)

        frame_video.append(frame.detach().cpu().numpy())
        mask_video.append(mask.detach().cpu().numpy())
        traj_video.append(traj_mask.detach().cpu().numpy())
        gt_mask_video.append(gt_mask)
        

def run_video_model(
    network,
    img_transform,
    mask_transform,
    frame_mask,
    phrase,
    sub_phrase,
    phrase_mask,
    sub_phrase_mask,
    image_files,
    mask_files,
    frame_video,
    mask_video,
    traj_video,
    gt_mask_video,
    num_frames = 16,
):
    video_queue = []
    for idx, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        
        image = Image.open(image_file).convert("RGB")
        gt_mask = Image.open(mask_file).convert("L")
        
        frame = img_transform(image) #.cuda(non_blocking=True).unsqueeze(0)
        gt_mask = mask_transform(gt_mask).unsqueeze(0) #.cuda(non_blocking=True).unsqueeze(0)
        
        if idx == 0:
            video_queue = [frame] * num_frames
        else:
            video_queue.pop()
            video_queue.append(frame)
        
        video_frames = torch.stack(video_queue, dim=1).cuda(non_blocking=True).unsqueeze(0)

        mask, traj_mask = network(video_frames, sub_phrase, frame_mask, sub_phrase_mask)

        frame_video.append(frame[None].detach().cpu().numpy())
        mask_video.append(mask[:, :, -1].detach().cpu().numpy())
        traj_video.append(traj_mask.detach().cpu().numpy())
        gt_mask_video.append(gt_mask)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VLN Navigation")

    parser.add_argument(
        "--data_root",
        default="/ssd_scratch/cvit/kanishk/carla_data",
        type=str,
        help="dataset name",
    )

    parser.add_argument(
        "--glove_path",
        default="/ssd_scratch/cvit/kanishk/glove",
        type=str,
        help="dataset name",
    )

    parser.add_argument(
        "--model",
        default="baseline",
        choices=["baseline"],
        type=str,
    )
    
    parser.add_argument(
        "--attn_type",
        default='dot_product',
        choices=[
            'dot_product',
            'scaled_dot_product',
            'multi_head',
            'rel_multi_head',
            'custom_attn'
        ],
        type=str,
    )

    parser.add_argument(
        "--img_backbone",
        default="vit_tiny_patch16_224",
        choices=[
            "vit_tiny_patch16_224",
            "vit_small_patch16_224",
            "vit_tiny_patch16_384",
            "vit_small_patch16_384",
            "dino_resnet50",
            "timesformer",
            "deeplabv3_resnet50",
            "deeplabv3_resnet101",
            "deeplabv3_mobilenet_v3_large",
            "convlstm"
        ],
        type=str,
    )

    parser.add_argument("--image_dim", type=int, default=448, help="Image Dimension")
    parser.add_argument("--mask_dim", type=int, default=448, help="Mask Dimension")
    parser.add_argument("--traj_dim", type=int, default=56, help="Trajk Dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden Dimension")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames of Video")
    parser.add_argument(
        "--patch_size", type=int, default=16, help="Patch Size of Video Frame for ViT"
    )

    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--threshold", type=float, default=0.4, help="mask threshold")

    parser.add_argument("--save", default=False, action="store_true")

    args = parser.parse_args()

    main(args)
