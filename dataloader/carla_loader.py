import os
import random
from random import sample
import re
from glob import glob
from collections import Iterable

import numpy as np
import pandas as pd

import cv2
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from einops import repeat

from .word_utils import Corpus

IGNORE = {
    "train": [
        "65",
        "100",
        "128",
        "153",
        "159",
        "163",
        "180",
        "197",
        "198",
        "204",
        "211",
        "221",
        "231",
        "233",
        "237",
        "242",
        "251",
        "263",
        "264",
        "265",
        "268",
        "27",
        "273",
        "278",
        "284",
        "286",
        "327",
        "356",
        "357",
        "366",
        "374",
        "390",
        "400",
        "402",
        "406",
        "413",
        "415",
        "429",
        "430",
        "437",
        "50",
        "75",
    ],
    "val": [],
}


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


def get_curve_length(points):
    return np.sum(
        [np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)]
    )


class CarlaFullDataset(Dataset):
    """Some Information about CarlaDataset"""

    def __init__(
        self,
        data_root,
        glove_path,
        split="train",
        img_transform=None,
        mask_transform=None,
        traj_transform=None,
        dataset_len=10000,
        skip=5,
        sequence_len=16,
        mode="image",
        one_in_n=1,
        image_dim=224,
        mask_dim=112,
        traj_dim=56,
        traj_frames=10,
        traj_size=25,
    ):
        self.data_dir = os.path.join(data_root, split)
        self.glove_path = glove_path
        self.split = split

        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.traj_transform = traj_transform

        self.dataset_len = dataset_len
        self.skip = skip
        self.sequence_len = sequence_len
        self.mode = mode
        self.one_in_n = one_in_n

        self.image_dim = image_dim
        self.mask_dim = mask_dim
        self.traj_dim = traj_dim

        self.traj_frames = traj_frames
        self.traj_size = traj_size

        if self.mode == "video":
            self.dataset_len = self.dataset_len // self.sequence_len

        self.episodes = sorted(os.listdir(self.data_dir))
        self.episodes = [episode for episode in self.episodes if episode.isnumeric()]
        print("Number of episodes before removal: ", len(self.episodes))

        for episode in IGNORE[split]:
            self.episodes.remove(episode)
        print("Number of episodes after removal: ", len(self.episodes))

        self.corpus = Corpus(glove_path)

        self.sub_command_data = pd.read_csv(
            f"./dataloader/sub_commands_{self.split}.csv", index_col=0
        )

        self.tree_embedding = torch.load(f"./dataloader/{self.split}_tree_embeddings.pt")

    def __len__(self):
        return self.dataset_len

    # TODO - Include Vehicle Position
    def get_video_data(
        self,
        episode_num,
        K,
        image_files,
        mask_files,
        matrix_files,
        vehicle_positions,
        target_positions,
        T=10,
    ):

        num_files = len(image_files)

        sample_idx = np.random.choice(range(num_files - self.skip - T))

        prev_idx = sample_idx
        while True:
            rgb_matrix = np.load(matrix_files[sample_idx])
            position_0 = vehicle_positions[sample_idx]
            position_0 = np.array(position_0).reshape(-1, 1)
            position_t = vehicle_positions[sample_idx + T // 2]
            position_t = np.array(position_t).reshape(-1, 1)

            pixel_t_2d = world_to_pixel(K, rgb_matrix, position_t, position_0)

            if (0 < pixel_t_2d[0] < 1280) and (0 < pixel_t_2d[1] < 720):
                break

            sample_idx += 1
            sample_idx %= num_files - T
            if prev_idx == sample_idx:
                print("remove ", image_files)
                break

        frames = []
        frame_masks = []
        orig_frames = []

        valid_indices = list(range(0, sample_idx, self.one_in_n))
        if len(valid_indices) > self.sequence_len:
            indices = valid_indices[-self.sequence_len :]
        else:
            indices = [0] * (self.sequence_len - len(valid_indices)) + valid_indices
        start_idx = indices[0]

        final_click_idx = target_positions["click_no"].max()

        sub_commands = []
        similarity_gts = []

        for index in indices:

            img_path = image_files[index]
            mask_path = mask_files[index]

            curr_click_idx = target_positions.iloc[index].to_list()[-1]

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            orig_frames.append(np.array(img))

            if self.img_transform:
                img = self.img_transform(img)

            if self.mask_transform:
                mask = self.mask_transform(mask)
            mask[mask > 0] = 1

            sub_command = self.tree_embedding[episode_num]["frame_sub_command"][index]
            similarity_gt = self.tree_embedding[episode_num]["frame_similarity_gt"][
                index
            ]

            frames.append(img)
            frame_masks.append(mask)
            sub_commands.append(sub_command)
            similarity_gts.append(similarity_gt)

        sub_phrases = self.tree_embedding[episode_num]["sub_phrases"]
        tree_embedding = self.tree_embedding[episode_num]["tree_embedding"]
        attention_mask = self.tree_embedding[episode_num]["attention_mask"]
        similarity_gts = torch.stack(similarity_gts, dim=0)

        orig_frames = np.stack(orig_frames, axis=0)
        frames = torch.stack(frames, dim=1)
        frame_masks = torch.stack(frame_masks, dim=1)

        rgb_matrix = np.load(matrix_files[sample_idx])

        pixel_coordinates = [np.array([0, 0])]
        position_0 = vehicle_positions[sample_idx]
        position_0 = np.array(position_0).reshape(-1, 1)

        for t in range(num_files - sample_idx - 1):
            position_t = vehicle_positions[sample_idx + t]
            position_t = np.array(position_t).reshape(-1, 1)

            pixel_t_2d = world_to_pixel(K, rgb_matrix, position_t, position_0)

            if pixel_t_2d.shape[-1] == 0:
                continue

            pixel_t_2d = np.array(
                [
                    int(pixel_t_2d[0]),
                    int(pixel_t_2d[1]),
                ]
            )
            diff = np.linalg.norm(pixel_t_2d - pixel_coordinates[-1])

            if diff > 20:
                pixel_coordinates.append(pixel_t_2d)

            if len(pixel_coordinates) > T:
                break

        pixel_coordinates = np.vstack(pixel_coordinates[1:])[:, None]

        traj_mask = np.zeros((orig_frames.shape[1], orig_frames.shape[2]))
        traj_mask = cv2.polylines(
            traj_mask,
            [pixel_coordinates],
            isClosed=False,
            color=(255),
            thickness=self.traj_size,
        )
        traj_mask = Image.fromarray(traj_mask)
        traj_mask = self.traj_transform(traj_mask)
        traj_mask[traj_mask > 0] = 1

        return (
            frames,
            orig_frames,
            frame_masks,
            traj_mask,
            sub_phrases,
            tree_embedding,
            attention_mask,
            similarity_gts,
            sample_idx,
        )

    def get_image_data(
        self,
        episode_num,
        K,
        image_files,
        mask_files,
        matrix_files,
        vehicle_positions,
        target_positions,
        T=10,
    ):

        num_files = len(image_files)

        sample_idx = np.random.choice(range(num_files - self.skip - T))

        prev_idx = sample_idx
        while True:
            rgb_matrix = np.load(matrix_files[sample_idx])
            position_0 = vehicle_positions[sample_idx]
            position_0 = np.array(position_0).reshape(-1, 1)
            position_t = vehicle_positions[sample_idx + T // 2]
            position_t = np.array(position_t).reshape(-1, 1)

            pixel_t_2d = world_to_pixel(K, rgb_matrix, position_t, position_0)

            if (0 < pixel_t_2d[0] < 1280) and (0 < pixel_t_2d[1] < 720):
                break

            sample_idx += 1
            sample_idx %= num_files - T
            if prev_idx == sample_idx:
                print("remove ", image_files)
                break

        img_path = image_files[sample_idx]
        mask_path = mask_files[sample_idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        final_click_idx = target_positions["click_no"].max()
        curr_click_idx = target_positions.iloc[sample_idx].to_list()[-1]

        orig_image = np.array(img)

        if self.img_transform:
            img = self.img_transform(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask[mask > 0] = 1

        mask_ = torch.zeros_like(mask)
        mask_ = repeat(mask_, "c h w -> (repeat c) h w", repeat=2)

        if curr_click_idx == final_click_idx:
            mask_[1] = mask[0]
        else:
            mask_[0] = mask[0]
        mask = mask_ + 1e-4

        rgb_matrix = np.load(matrix_files[sample_idx])

        pixel_coordinates = [np.array([0, 0])]
        position_0 = vehicle_positions[sample_idx]
        position_0 = np.array(position_0).reshape(-1, 1)

        for t in range(num_files - sample_idx - 1):
            position_t = vehicle_positions[sample_idx + t]
            position_t = np.array(position_t).reshape(-1, 1)

            pixel_t_2d = world_to_pixel(K, rgb_matrix, position_t, position_0)

            if pixel_t_2d.shape[-1] == 0:
                continue

            pixel_t_2d = np.array(
                [
                    int(pixel_t_2d[0]),
                    int(pixel_t_2d[1]),
                ]
            )
            diff = np.linalg.norm(pixel_t_2d - pixel_coordinates[-1])

            if diff > 20:
                pixel_coordinates.append(pixel_t_2d)

            if len(pixel_coordinates) > T:
                break

        pixel_coordinates = np.vstack(pixel_coordinates[1:])[:, None]

        traj_mask = np.zeros((orig_image.shape[0], orig_image.shape[1]))
        traj_mask = cv2.polylines(
            traj_mask,
            [pixel_coordinates],
            isClosed=False,
            color=(255),
            thickness=self.traj_size,
        )
        traj_mask = Image.fromarray(traj_mask)
        traj_mask = self.traj_transform(traj_mask)
        traj_mask[traj_mask > 0] = 1

        return img, orig_image, mask, traj_mask, sample_idx

    def __getitem__(self, idx):
        output = {}

        episode = np.random.choice(self.episodes)
        episode_dir = os.path.join(self.data_dir, episode)

        # import pdb; pdb.set_trace()
        episode_num = int(episode.split("/")[-1])

        image_files = sorted(glob(episode_dir + f"/images/*.png"))
        mask_files = sorted(glob(episode_dir + f"/masks/*.png"))
        matrix_files = sorted(glob(episode_dir + f"/inverse_matrix/*.npy"))
        position_file = os.path.join(episode_dir, "vehicle_positions.txt")
        command_path = os.path.join(episode_dir, "command.txt")
        intrinsic_path = os.path.join(episode_dir, "camera_intrinsic.npy")
        target_path = os.path.join(episode_dir, "target_positions.txt")

        K = np.load(intrinsic_path)

        vehicle_positions = []
        with open(position_file, "r") as fhand:
            for line in fhand:
                position = np.array(line.split(","), dtype=np.float32)
                vehicle_positions.append(position)

        target_positions = pd.read_csv(target_path, names=["x", "y", "z", "click_no"])

        traj_mask = None
        sample_idx = None
        if self.mode == "image":
            (
                frames,
                orig_frames,
                frame_mask,
                traj_mask,
                sub_commands,
                sample_idx,
            ) = self.get_image_data(
                episode_num,
                K,
                image_files,
                mask_files,
                matrix_files,
                vehicle_positions,
                target_positions,
                T=self.traj_frames,
            )
        elif self.mode == "video":
            (
                frames,
                orig_frames,
                frame_mask,
                traj_mask,
                sub_phrases,
                tree_embedding,
                attention_mask,
                similarity_gts,
                sample_idx,
            ) = self.get_video_data(
                episode_num,
                K,
                image_files,
                mask_files,
                matrix_files,
                vehicle_positions,
                target_positions,
                T=self.traj_frames,
            )
        else:
            raise NotImplementedError(f"{self.mode} mode not implemented!")

        output["orig_frame"] = orig_frames
        output["frame"] = frames
        output["gt_frame"] = frame_mask
        output["gt_traj_mask"] = traj_mask
        output["episode"] = episode_dir.split("/")[-1]
        output["sample_idx"] = sample_idx

        command = open(command_path, "r").read()
        command = self.sub_command_data.loc[episode_num]["command"]
        command = re.sub(r"[^\w\s]", "", command)
        # tokens, phrase_mask = self.corpus.tokenize(command)

        # import pdb; pdb.set_trace()
        # output["sub_phrases"] = sub_phrases
        # output["tree_embedding"] = tree_embedding
        # output["attention_mask"] = attention_mask
        # output["similarity_gts"] = similarity_gts

        positive_anchor = tree_embedding[similarity_gts.argmax()]
        positive_anchor = torch.stack([positive_anchor]*self.sequence_len, dim=0)
        positive_anchor_mask = attention_mask[similarity_gts.argmax()]
        
        # negative_indices = torch.where(similarity_gts == -1)
        # negative_index = random.choice(negative_indices)
        # negative_anchor = tree_embedding[negative_index]
        
        negative_anchor = []
        negative_anchor_mask = []
        for similarity_gt in similarity_gts:
            negative_indices = torch.where(similarity_gt == -1)[0]
            negative_index = random.choice(negative_indices)
            negative_anchor.append(tree_embedding[negative_index])
            negative_anchor_mask.append(attention_mask[negative_index])
        negative_anchor = torch.stack(negative_anchor, dim=0)
        negative_anchor_mask = torch.stack(negative_anchor_mask, dim=0)
        
        output["positive_anchor"] = positive_anchor
        output["positive_anchor_mask"] = positive_anchor_mask
        
        output["negative_anchor"] = negative_anchor
        output["negative_anchor_mask"] = negative_anchor_mask
        
        output["anchor"] = frames
        output["anchor_mask"] = frame_mask
        output["gt_traj_mask"] = traj_mask

        for key in output:
            if torch.is_tensor(output[key]):
                print(output[key].shape)
        
        return output
