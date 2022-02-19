import wandb

import re
import os
import argparse
from glob import glob

import numpy as np
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

    val_path = os.path.join(args.data_root, 'val/')
    glove_path = args.glove_path
    checkpoint_path = './saved_model/baseline_deeplabv3_resnet50_18_Feb_06-11.pth'

    corpus = Corpus(glove_path)

    return_layers = {"layer2": "layer2",
                     "layer3": "layer3", "layer4": "layer4"}

    mode = "image"
    if "vit_" in args.img_backbone:
        img_backbone = timm.create_model(args.img_backbone, pretrained=True)
        visual_encoder = nn.Sequential(*list(img_backbone.children())[:-1])
        network = SegmentationBaseline(
            visual_encoder,
            hidden_dim=args.hidden_dim,
            mask_dim=args.mask_dim,
            backbone=args.img_backbone,
        )
    elif "dino_resnet50" in args.img_backbone:
        img_backbone = torch.hub.load(
            "facebookresearch/dino:main", "dino_resnet50")
        visual_encoder = IntermediateLayerGetter(img_backbone, return_layers)
        network = IROSBaseline(
            visual_encoder, hidden_dim=args.hidden_dim, mask_dim=args.mask_dim
        )
    elif "timesformer" in args.img_backbone:
        mode = "video"
        spatial_dim = args.image_dim//args.patch_size
        visual_encoder = VisionTransformer(img_size=args.image_dim, patch_size=args.patch_size,
                                           embed_dim=args.hidden_dim, depth=2, num_heads=8, num_frames=args.num_frames)
        network = VideoSegmentationBaseline(
            visual_encoder, hidden_dim=args.hidden_dim, mask_dim=args.mask_dim, spatial_dim=spatial_dim, num_frames=args.num_frames,
        )
    elif "deeplabv3_" in args.img_backbone:
        img_backbone = torch.hub.load(
            "pytorch/vision:v0.10.0", args.img_backbone, pretrained=True
        )
        visual_encoder = nn.Sequential(
            *list(img_backbone._modules["backbone"].children())
        )
        network = SegmentationBaseline(
            visual_encoder,
            hidden_dim=args.hidden_dim,
            mask_dim=args.mask_dim,
            backbone=args.img_backbone,
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

    frame_mask = torch.ones(
        1, 14 * 14, dtype=torch.int64).cuda(non_blocking=True)

    columns = ["Command"]

    episodes = glob(val_path + "*")
    episode_num = 0
    for episode in episodes:

        command_path = episode + "/command.txt"
        command = open(command_path, "r").read()
        command = re.sub(r"[^\w\s]", "", command)

        phrase, phrase_mask = corpus.tokenize(command)
        phrase = phrase.unsqueeze(0)
        phrase_mask = phrase_mask.unsqueeze(0)

        image_files = sorted(glob(episode + "/images/*.png"))
        mask_files = sorted(glob(episode + "/masks/*.png"))

        frame_video = []
        mask_video = []
        gt_mask_video = []

        for image_file, mask_file in zip(image_files, mask_files):

            image = Image.open(image_file).convert("RGB")
            gt_mask = Image.open(mask_file).convert("L")

            frame = img_transform(image).cuda(non_blocking=True).unsqueeze(0)
            gt_mask = mask_transform(gt_mask).cuda(
                non_blocking=True).unsqueeze(0)

            mask = network(frame, phrase, frame_mask, phrase_mask)

            frame_video.append(frame.detach().cpu().numpy())
            mask_video.append(mask.detach().cpu().numpy())
            gt_mask_video.append(gt_mask.detach().cpu().numpy())

        frame_video = np.concatenate(frame_video, axis=0)
        mask_video = np.concatenate(mask_video, axis=0)
        gt_mask_video = np.concatenate(gt_mask_video, axis=0)

        # import pdb; pdb.set_trace()
        mask_video_overlay = np.copy(frame_video)
        mask_video_overlay[:, 0] += (mask_video[:, 0]/mask_video.max())
        mask_video_overlay = np.clip(mask_video_overlay, a_min=0., a_max=1.)

        gt_mask_video_overlay = np.copy(frame_video)
        gt_mask_video_overlay[:, 0] += gt_mask_video[:, 0]
        gt_mask_video_overlay = np.clip(
            gt_mask_video_overlay, a_min=0., a_max=1.)

        frame_video = np.uint8(frame_video * 255)
        mask_video = np.uint8(mask_video_overlay * 255)
        gt_mask_video = np.uint8(gt_mask_video_overlay * 255)

        print(episode_num, command)

        wandb.log(
            {
                "video": wandb.Video(frame_video, fps=4, caption=command, format="mp4"),
                "pred_mask": wandb.Video(mask_video, fps=4, caption=command, format="mp4"),
                "gt_mask": wandb.Video(gt_mask_video, fps=4, caption=command, format="mp4"),
            }
        )

        episode_num += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VLN Navigation")
    parser.add_argument("--seed", default=420, type=int, help="random seed")
    parser.add_argument("--epochs", default=200, type=int, help="epoch size")

    parser.add_argument("--batch_size", default=100,
                        type=int, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=10, help="number of workers")

    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--gamma", default=0.7, type=float)
    parser.add_argument(
        "--optimizer",
        default="AdamW",
        choices=["AdamW", "Adam", "SGD", "RMSprop", "Rprop", "ASGD", "RAdam"],
        type=str,
    )

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
        default='baseline',
        choices=[
            'baseline'
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
            "deeplabv3_mobilenet_v3_large"
        ],
        type=str,
    )

    parser.add_argument("--image_dim", type=int,
                        default=448, help="Image Dimension")
    parser.add_argument("--mask_dim", type=int,
                        default=448, help="Mask Dimension")
    parser.add_argument("--hidden_dim", type=int,
                        default=256, help="Hidden Dimension")
    parser.add_argument("--num_frames", type=int,
                        default=16, help="Frames of Video")
    parser.add_argument("--patch_size", type=int,
                        default=16, help="Patch Size of Video Frame for ViT")

    parser.add_argument("--grad_check", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default="./saved_model")

    parser.add_argument("--threshold", type=float,
                        default=0.4, help="mask threshold")

    parser.add_argument("--save", default=False, action="store_true")

    args = parser.parse_args()

    main(args)
