import argparse
import os
import re
from glob import glob
from cv2 import threshold

import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.optim import *
from torchvision.models._utils import IntermediateLayerGetter

from dataloader.carla_loader import *
from dataloader.word_utils import Corpus
from models.model import *
from timesformer.models.vit import VisionTransformer
from utilities.loss import *
from utilities.metrics import *
from utilities.utilities import *


def world_to_pixel(K, rgb_matrix, destination, curr_position):
    point_3d = np.ones((4, destination.shape[1]))
    point_3d[0] = destination[0]
    point_3d[1] = destination[1]
    point_3d[2] = curr_position[2]

    point_3d = np.round(point_3d, decimals=2)
    cam_coords = rgb_matrix @ point_3d
    cam_coords = np.array([cam_coords[1], cam_coords[2] * -1, cam_coords[0]])

    points_2d = np.dot(K, cam_coords)

    points_2d = np.array(
        [
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :],
        ]
    )
    points_2d = points_2d.reshape(3, -1)
    points_2d = np.round(points_2d, decimals=2)
    return points_2d


def get_traj_mask(
    num_files,
    cur_file,
    orig_image,
    K,
    traj_transform,
    matrix_files,
    vehicle_positions,
    traj_frames=10,
):
    rgb_matrix = np.load(matrix_files[cur_file])

    pixel_coordinates = [np.array([0, 0])]
    position_0 = vehicle_positions[cur_file]
    position_0 = np.array(position_0).reshape(-1, 1)

    for t in range(cur_file, num_files):
        position_t = vehicle_positions[t]
        position_t = np.array(position_t).reshape(-1, 1)

        pixel_t_2d = world_to_pixel(K, rgb_matrix, position_t, position_0)

        pixel_t_2d = np.array([int(pixel_t_2d[0]), int(pixel_t_2d[1]),])
        pixel_coordinates.append(pixel_t_2d)

        if len(pixel_coordinates) == traj_frames:
            break

    pixel_coordinates = np.vstack(pixel_coordinates[1:])[:, None]
    traj_mask = np.zeros((orig_image.shape[0], orig_image.shape[1]))
    traj_mask = cv2.polylines(
        traj_mask, [pixel_coordinates], isClosed=False, color=(255), thickness=25
    )

    traj_mask = Image.fromarray(traj_mask)
    traj_mask = traj_transform(traj_mask)
    traj_mask[traj_mask > 0] = 1

    return traj_mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpu = torch.cuda.device_count()
    print(f"Using {device} with {num_gpu} GPUS!")

    val_path = os.path.join(args.data_root, "val/")
    glove_path = args.glove_path
    checkpoint_path = args.checkpoint
    threshold = args.threshold

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
        spatial_dim = args.image_dim // args.patch_size
        video_encoder = torch.hub.load(
            "facebookresearch/pytorchvideo", "x3d_s", pretrained=True
        )
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
    elif "conv3d_baseline" in args.img_backbone:
        mode = "video"
        spatial_dim = args.image_dim // args.patch_size

        video_encoder = torch.hub.load(
            "facebookresearch/pytorchvideo", "x3d_s", pretrained=True
        )
        visual_encoder = nn.Sequential(*list(video_encoder.blocks.children())[:-1])

        network = Conv3D_Baseline(
            visual_encoder,
            hidden_dim=args.hidden_dim,
            image_dim=args.image_dim,
            mask_dim=args.mask_dim,
            traj_dim=args.traj_dim,
            spatial_dim=spatial_dim,
            num_frames=args.num_frames,
            attn_type=args.attn_type,
        )

    if num_gpu > 1:
        network = nn.DataParallel(network)
        print("Using DataParallel mode!")
    network.to(device)

    checkpoint = torch.load(checkpoint_path)
    network.load_state_dict(checkpoint["state_dict"])

    network.eval()

    img_transform = transforms.Compose(
        [transforms.Resize((args.image_dim, args.image_dim)), transforms.ToTensor(),]
    )

    mask_transform = transforms.Compose(
        [transforms.Resize((args.mask_dim, args.mask_dim)), transforms.ToTensor(),]
    )

    traj_transform = transforms.Compose(
        [transforms.Resize((args.traj_dim, args.traj_dim)), transforms.ToTensor(),]
    )

    frame_mask = torch.ones(1, 7 * 7, dtype=torch.int64).cuda(non_blocking=True)

    total_inter_traj, total_union_traj = 0, 0
    total_rk_traj = {1: 0, 10: 0, 100: 0, 1000: 0}
    total_pg_traj = 0

    total_inter_mask, total_union_mask = 0, 0
    total_rk_mask = {1: 0, 10: 0, 100: 0, 1000: 0}
    total_pg_mask = 0

    episodes = glob(val_path + "*")
    episode_num = 0
    for episode in episodes:
        episode_num = int(episode.split("/")[-1])

        command_path = os.path.join(episode, "command.txt")

        command = open(command_path, "r").read()
        command = re.sub(r"[^\w\s]", "", command)

        phrase, phrase_mask = corpus.tokenize(command)
        phrase = phrase.cuda(non_blocking=True).unsqueeze(0)
        phrase_mask = phrase_mask.cuda(non_blocking=True).unsqueeze(0)

        sub_phrase = torch.stack([phrase] * args.num_frames, dim=1)
        sub_phrase_mask = torch.stack([phrase_mask] * args.num_frames, dim=1)

        image_files = sorted(glob(episode + f"/images/*.png"))
        mask_files = sorted(glob(episode + f"/masks/*.png"))
        matrix_files = sorted(glob(episode + f"/inverse_matrix/*.npy"))
        position_file = os.path.join(episode, "vehicle_positions.txt")
        intrinsic_path = os.path.join(episode, "camera_intrinsic.npy")

        K = np.load(intrinsic_path)
        vehicle_positions = []
        with open(position_file, "r") as fhand:
            for line in fhand:
                position = np.array(line.split(","), dtype=np.float32)
                vehicle_positions.append(position)

        frame_video = []
        mask_video = []
        traj_video = []
        gt_mask_video = []
        gt_traj_video = []

        if mode == "image":
            run_image_model(
                network,
                img_transform,
                mask_transform,
                traj_transform,
                frame_mask,
                phrase,
                sub_phrase,
                phrase_mask,
                sub_phrase_mask,
                image_files,
                mask_files,
                matrix_files,
                K,
                vehicle_positions,
                frame_video,
                mask_video,
                traj_video,
                gt_mask_video,
                gt_traj_video,
            )
        else:
            run_video_model(
                network,
                img_transform,
                mask_transform,
                traj_transform,
                frame_mask,
                phrase,
                sub_phrase,
                phrase_mask,
                sub_phrase_mask,
                image_files,
                mask_files,
                matrix_files,
                K,
                vehicle_positions,
                frame_video,
                mask_video,
                traj_video,
                gt_mask_video,
                gt_traj_video,
                num_frames=args.num_frames,
            )

        frame_video = np.concatenate(frame_video, axis=0)
        mask_video = np.concatenate(mask_video, axis=0)
        traj_video = np.concatenate(traj_video, axis=0)
        gt_mask_video = np.concatenate(gt_mask_video, axis=0)

        inter_traj, union_traj = compute_mask_IOU(
            torch.from_numpy(np.asarray(traj_video)),
            torch.from_numpy(np.asarray(gt_traj_video)),
            threshold,
        )

        total_inter_traj += inter_traj.item()
        total_union_traj += union_traj.item()

        total_pg_traj += pointing_game(
            torch.from_numpy(np.asarray(traj_video)),
            torch.from_numpy(np.asarray(gt_traj_video)),
        )
        for k in total_rk_traj:
            total_rk_traj[k] += recall_at_k(
                torch.from_numpy(np.asarray(traj_video)),
                torch.from_numpy(np.asarray(gt_traj_video)),
                topk=k,
            )

        # inter_mask, union_mask = compute_mask_IOU(
        #     torch.from_numpy(np.asarray(mask_video)),
        #     torch.from_numpy(np.asarray(gt_mask_video)),
        #     threshold,
        # )

        # total_inter_mask += inter_mask.item()
        # total_union_mask += union_mask.item()

        # total_pg_mask += pointing_game(
        #     torch.from_numpy(np.asarray(mask_video)),
        #     torch.from_numpy(np.asarray(gt_mask_video)),
        # )
        # for k in total_rk_mask:
        #     total_rk_mask[k] += recall_at_k(
        #         torch.from_numpy(np.asarray(mask_video)),
        #         torch.from_numpy(np.asarray(gt_mask_video)),
        #         topk=k,
        #     )

        episode_num += 1

    val_IOU_traj = total_inter_traj / total_union_traj
    val_pg_traj = total_pg_traj / episode_num
    val_rk_traj = {}
    for k in total_rk_traj:
        val_rk_traj[k] = total_rk_traj[k] / episode_num

    print(
        f"Traj_IOU {val_IOU_traj:.4f} Traj_PG {val_pg_traj:.4f} Traj RK {val_rk_traj[1]:.4f} (k = 1), {val_rk_traj[10]:.4f} (k = 10), {val_rk_traj[100]:.4f} (k = 100), {val_rk_traj[1000]:.4f} (k = 1000)"
    )

    # val_IOU_mask = total_inter_mask / total_union_mask
    # val_pg_mask = total_pg_mask / episode_num
    # val_rk_mask = {}
    # for k in total_rk_mask:
    #     val_rk_mask[k] = total_rk_mask[k] / episode_num

    # print(
    #     f"Mask_IOU {val_IOU_mask:.4f} Mask_PG {val_pg_mask:.4f} Mask RK {val_rk_mask[1]:.4f} (k = 1), {val_rk_mask[10]:.4f} (k = 10), {val_rk_mask[100]:.4f} (k = 100), {val_rk_mask[1000]:.4f} (k = 1000)"
    # )


def run_image_model(
    network,
    img_transform,
    mask_transform,
    traj_transform,
    frame_mask,
    phrase,
    sub_phrase,
    phrase_mask,
    sub_phrase_mask,
    image_files,
    mask_files,
    matrix_files,
    K,
    vehicle_positions,
    frame_video,
    mask_video,
    traj_video,
    gt_mask_video,
    gt_traj_video,
):
    num_files = len(image_files)
    traj_frames = 10
    cur_file = 0
    for image_file, mask_file in zip(image_files, mask_files):
        image = Image.open(image_file).convert("RGB")
        gt_mask = Image.open(mask_file).convert("L")

        frame = img_transform(image).cuda(non_blocking=True).unsqueeze(0)
        gt_mask = mask_transform(gt_mask).unsqueeze(0)

        mask, traj_mask = network(frame, phrase, frame_mask, phrase_mask)

        gt_traj = get_traj_mask(
            num_files,
            cur_file,
            np.array(image),
            K,
            traj_transform,
            matrix_files,
            vehicle_positions,
            traj_frames=traj_frames,
        )

        frame_video.append(frame.detach().cpu().numpy())
        mask_video.append(mask.detach().cpu().numpy())
        traj_video.append(traj_mask.detach().cpu().numpy())
        gt_mask_video.append(gt_mask.detach().cpu().numpy())
        gt_traj_video.append(gt_traj.detach().cpu().numpy())

        cur_file += 1
        if cur_file + traj_frames == num_files:
            break


def run_video_model(
    network,
    img_transform,
    mask_transform,
    traj_transform,
    frame_mask,
    phrase,
    sub_phrase,
    phrase_mask,
    sub_phrase_mask,
    image_files,
    mask_files,
    matrix_files,
    K,
    vehicle_positions,
    frame_video,
    mask_video,
    traj_video,
    gt_mask_video,
    gt_traj_video,
    num_frames=16,
):
    num_files = len(image_files)
    traj_frames = 10
    cur_file = 0
    video_queue = []
    for idx, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):

        image = Image.open(image_file).convert("RGB")
        gt_mask = Image.open(mask_file).convert("L")

        frame = img_transform(image)
        gt_mask = mask_transform(gt_mask).unsqueeze(0)

        if idx == 0:
            video_queue = [frame] * num_frames
        else:
            video_queue.pop()
            video_queue.append(frame)

        video_frames = (
            torch.stack(video_queue, dim=1).cuda(non_blocking=True).unsqueeze(0)
        )

        mask, traj_mask = network(video_frames, phrase, frame_mask, phrase_mask)
        gt_traj = get_traj_mask(
            num_files,
            cur_file,
            np.array(image),
            K,
            traj_transform,
            matrix_files,
            vehicle_positions,
            traj_frames=traj_frames,
        )

        frame_video.append(frame.detach().cpu().numpy())
        mask_video.append(mask.detach().cpu().numpy())
        traj_video.append(traj_mask.detach().cpu().numpy())
        gt_mask_video.append(gt_mask.detach().cpu().numpy())
        gt_traj_video.append(gt_traj.detach().cpu().numpy())

        cur_file += 1
        if cur_file + traj_frames == num_files:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLN Navigation")

    parser.add_argument(
        "--data_root",
        default="/scratch/ashwin_mittal/dataset",
        type=str,
        help="dataset name",
    )

    parser.add_argument(
        "--glove_path",
        default="/scratch/ashwin_mittal/glove",
        type=str,
        help="dataset name",
    )

    parser.add_argument(
        "--model", default="baseline", choices=["baseline"], type=str,
    )

    parser.add_argument(
        "--attn_type",
        default="dot_product",
        choices=[
            "dot_product",
            "scaled_dot_product",
            "multi_head",
            "rel_multi_head",
            "custom_attn",
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
            "convlstm",
            "conv3d_baseline",
        ],
        type=str,
    )

    parser.add_argument("--image_dim", type=int, default=448, help="Image Dimension")
    parser.add_argument("--mask_dim", type=int, default=448, help="Mask Dimension")
    parser.add_argument("--traj_dim", type=int, default=56, help="Traj Dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden Dimension")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames of Video")
    parser.add_argument(
        "--patch_size", type=int, default=16, help="Patch Size of Video Frame for ViT"
    )

    parser.add_argument(
        "--checkpoint",
        default="/scratch/ashwin_mittal/models/conv3d_baseline_class_level_combo_multi_head_hd_384_sf_10_tf_20_05_Apr_09_00.pth",
        type=str,
    )

    parser.add_argument("--threshold", type=float, default=0.4, help="mask threshold")

    parser.add_argument("--save", default=False, action="store_true")

    args = parser.parse_args()

    main(args)

    # /home2/ashwin_mittal/miniconda3/bin/python /home2/ashwin_mittal/Validation/carla_nav/validation.py --image_dim 224 --mask_dim 112 --traj_dim 56 --img_backbone vit_small_patch16_224 --hidden_dim 384 --num_frames 4 --attn_type multi_head
    # /home2/ashwin_mittal/miniconda3/bin/python /home2/ashwin_mittal/Validation/carla_nav/validation.py --image_dim 224 --mask_dim 112 --traj_dim 56 --img_backbone conv3d_baseline --hidden_dim 384 --num_frames 4 --attn_type multi_head
