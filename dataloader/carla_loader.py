import os
import re
from glob import glob
from collections import Iterable

import numpy as np
import torch

import cv2
from PIL import Image

from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .word_utils import Corpus

def world_to_pixel(K, rgb_matrix, destination,  curr_position):
    point_3d = np.ones((4, destination.shape[1]))
    point_3d[0] = destination[0]
    point_3d[1] = destination[1]
    point_3d[2] = curr_position[2]

    # point_3d = np.array([destination[0], destination[1], curr_position[2], 1])
    point_3d = np.round(point_3d, decimals=2)
    # print("3D world coordinate: ", point_3d)

    cam_coords = rgb_matrix @ point_3d
    # cam_coords = rgb_matrix @ point_3d[:, None]
    cam_coords = np.array([cam_coords[1], cam_coords[2]*-1, cam_coords[0]])

    # cam_coords = cam_coords[:, cam_coords[2, :] > 0]
    # cam_coords[2] = abs(cam_coords[2])
    points_2d = np.dot(K, cam_coords)

    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]]
    )
    points_2d = points_2d.reshape(3, -1)
    points_2d = np.round(points_2d, decimals=2)
    return points_2d


def get_curve_length(points):
    return np.sum([np.linalg.norm(points[i + 1] - points[i])  for i in range(len(points) - 1)])


class CarlaDataset(Dataset):
    """Some Information about CarlaDataset"""

    def __init__(
        self,
        data_root,
        glove_path,
        split="train",
        img_transform=None,
        mask_transform=None,
        dataset_len=10000,
        skip=10,
        sequence_len=16,
        mode="image",
        image_dim=224, 
        mask_dim=112
    ):
        self.data_dir = os.path.join(data_root, split)

        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.dataset_len = dataset_len
        self.skip = skip
        self.sequence_len = sequence_len
        self.mode = mode
        
        self.image_dim = image_dim
        self.mask_dim = mask_dim

        if self.mode == "video":
            self.dataset_len = self.dataset_len//self.sequence_len

        self.episodes = sorted(os.listdir(self.data_dir))

        self.corpus = Corpus(glove_path)

    def __len__(self):
        return self.dataset_len

    def get_video_data(self, image_files, mask_files, num_files):
        sample_idx = np.random.choice(range(num_files - self.sequence_len))

        frames = []
        orig_frames = []
        frame_masks = []

        for index in range(self.sequence_len):
            img_path = image_files[sample_idx + index]
            mask_path = mask_files[sample_idx + index]

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            orig_frames.append(np.array(img))

            if self.img_transform:
                img = self.img_transform(img)

            if self.mask_transform:
                mask = self.mask_transform(mask)
                mask[mask > 0] = 1

            frames.append(img)
            frame_masks.append(mask)

        orig_frames = np.stack(orig_frames, axis=0)
        frames = torch.stack(frames, dim=1)
        frame_masks = torch.stack(frame_masks, dim=1)
        return frames, orig_frames[-1], frame_masks[:, -1]

    def get_image_data(self, image_files, mask_files, num_files):
        sample_idx = np.random.choice(range(num_files - self.skip))

        img_path = image_files[sample_idx]
        mask_path = mask_files[sample_idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        orig_image = np.array(img)

        if self.img_transform:
            img = self.img_transform(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask[mask > 0] = 1
        return img, orig_image, mask

    def __getitem__(self, idx):
        output = {}

        episode_dir = os.path.join(self.data_dir, np.random.choice(self.episodes))

        image_files = sorted(glob(episode_dir + f"/images/*.png"))
        mask_files = sorted(glob(episode_dir + f"/masks/*.png"))
        command_path = os.path.join(episode_dir, "command.txt")

        num_files = len(image_files)

        if self.mode == "image":
            frames, orig_frames, frame_masks = self.get_image_data(
                image_files, mask_files, num_files
            )
        elif self.mode == "video":
            frames, orig_frames, frame_masks = self.get_video_data(
                image_files, mask_files, num_files
            )
        else:
            raise NotImplementedError(f"{self.mode} mode not implemented!")

        output["orig_frame"] = orig_frames
        output["frame"] = frames
        output["gt_frame"] = frame_masks

        command = open(command_path, "r").read()
        command = re.sub(r"[^\w\s]", "", command)
        output["orig_text"] = command

        tokens, phrase_mask = self.corpus.tokenize(output["orig_text"])
        output["text"] = tokens
        output["text_mask"] = phrase_mask

        return output


class CarlaFullDataset(Dataset):
    """Some Information about CarlaDataset"""

    def __init__(
        self,
        data_root,
        glove_path,
        split="train",
        img_transform=None,
        mask_transform=None,
        dataset_len=10000,
        skip=20,
        sequence_len=16,
        mode="image",
        image_dim=224, 
        mask_dim=112
    ):
        self.data_dir = os.path.join(data_root, split)

        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.dataset_len = dataset_len
        self.skip = skip
        self.sequence_len = sequence_len
        self.mode = mode
        
        self.image_dim = image_dim
        self.mask_dim = mask_dim

        if self.mode == "video":
            self.dataset_len = self.dataset_len//self.sequence_len

        self.episodes = sorted(os.listdir(self.data_dir))

        self.corpus = Corpus(glove_path)

    def __len__(self):
        return self.dataset_len

    # TODO - Include Vehicle Position
    def get_video_data(self, image_files, mask_files):
        
        num_files = len(image_files)
        
        sample_idx = np.random.choice(range(num_files - self.sequence_len))

        frames = []
        orig_frames = []
        frame_masks = []

        for index in range(self.sequence_len):
            img_path = image_files[sample_idx + index]
            mask_path = mask_files[sample_idx + index]

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            orig_frames.append(np.array(img))

            if self.img_transform:
                img = self.img_transform(img)

            if self.mask_transform:
                mask = self.mask_transform(mask)
                mask[mask > 0] = 1

            frames.append(img)
            frame_masks.append(mask)

        orig_frames = np.stack(orig_frames, axis=0)
        frames = torch.stack(frames, dim=1)
        frame_masks = torch.stack(frame_masks, dim=1)
        return frames, orig_frames[-1], frame_masks[:, -1]
    
    # TODO - Include Vehicle Position
    # Convert the current position and next position to pixel coordinates 
    # using the current camera transformation matrix 
    # the coordinates should be rescaled (original image resolution to resized image resolution) and normalized
    def get_image_data(self, K, image_files, mask_files, matrix_files, vehicle_positions, T=10):
        
        num_files = len(image_files)
        
        sample_idx = np.random.choice(range(num_files - self.skip - T))
        
        prev_idx = sample_idx
        while True:
            rgb_matrix = np.load(matrix_files[sample_idx])
            position_0 = vehicle_positions[sample_idx]
            position_0 = np.array(position_0).reshape(-1, 1)
            position_t = vehicle_positions[sample_idx + T // 2]
            position_t = np.array(position_t).reshape(-1, 1)

            # Convert the current position and next position to pixel coordinates
            # using the current camera transformation matrix
            pixel_t_2d = world_to_pixel(K, rgb_matrix, position_t, position_0)

            if (
                pixel_t_2d[0] > 0
                and pixel_t_2d[0] < 1280
                and pixel_t_2d[1] > 0
                and pixel_t_2d[1] < 720
            ):
                break

            sample_idx += 1
            sample_idx %= num_files - self.skip - T
            if prev_idx == sample_idx:
                print(matrix_files[sample_idx])

        # train -> 109 113 114 121 128 131 132 140 146 15 152 155 156 159 161 166 171 172 177 179 19 195 206 214 215 216 222 230 27 30 31 34 35 49 58 59 61 68 7 72 73 74 81 82 83 86 88 91 92 96 98 54
        # val -> 1 11 14 18 2 25 28 32 33 34 37 39 44 46 5 50 7 
        
        img_path = image_files[sample_idx]
        mask_path = mask_files[sample_idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        orig_image = np.array(img)

        if self.img_transform:
            img = self.img_transform(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask[mask > 0] = 1
        
        rgb_matrix = np.load(matrix_files[sample_idx])
        
        pixel_coordinates = [np.array([0, 0])]
        position_0 = vehicle_positions[sample_idx]
        position_0 = np.array(position_0).reshape(-1, 1)

        for t in range(num_files - sample_idx - 1):
            position_t = vehicle_positions[sample_idx + t]
            position_t = np.array(position_t).reshape(-1, 1)

            # Convert the current position and next position to pixel coordinates
            # using the current camera transformation matrix
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
            # print(t, diff)

            if diff > 20:
                pixel_coordinates.append(pixel_t_2d)

            if len(pixel_coordinates) > T:
                break
        
        pixel_coordinates = np.vstack(pixel_coordinates[1:])[:, None]
        
        traj_mask = np.zeros((orig_image.shape[0], orig_image.shape[1]))
        traj_mask = cv2.polylines(traj_mask, [pixel_coordinates], isClosed=False, color=(255), thickness=25)
        traj_mask = Image.fromarray(traj_mask)
        traj_mask = self.mask_transform(traj_mask)
        traj_mask[traj_mask > 0] = 1

        return img, orig_image, mask, traj_mask, sample_idx

    def __getitem__(self, idx):
        output = {}

        episode_dir = os.path.join(self.data_dir, np.random.choice(self.episodes))

        image_files = sorted(glob(episode_dir + f"/images/*.png"))
        mask_files = sorted(glob(episode_dir + f"/masks/*.png"))
        matrix_files = sorted(glob(episode_dir + f"/inverse_matrix/*.npy"))
        position_file = os.path.join(episode_dir, "vehicle_positions.txt")
        command_path = os.path.join(episode_dir, "command.txt")
        intrinsic_path = os.path.join(episode_dir, "camera_intrinsic.npy")

        K = np.load(intrinsic_path)

        vehicle_positions = []
        with open(position_file, "r") as fhand:
            for line in fhand:
                position = np.array(line.split(","), dtype=np.float32)
                vehicle_positions.append(position)
                
        # print(len(image_files), len(mask_files), len(matrix_files), len(vehicle_positions), episode_dir)
        # assert len(image_files) == len(mask_files) == len(matrix_files) == len(vehicle_positions)
            
        traj_mask = None
        sample_idx = None
        if self.mode == "image":
            frames, orig_frames, frame_masks, traj_mask, sample_idx = self.get_image_data(
                K, image_files, mask_files, matrix_files, vehicle_positions
            )
        elif self.mode == "video":
            frames, orig_frames, frame_masks = self.get_video_data(
                image_files, mask_files
            )
        else:
            raise NotImplementedError(f"{self.mode} mode not implemented!")

        output["orig_frame"] = orig_frames
        output["frame"] = frames
        output["gt_frame"] = frame_masks
        output["gt_traj_mask"] = traj_mask
        output["episode"] = episode_dir.split("/")[-1]
        output["sample_idx"] = sample_idx

        command = open(command_path, "r").read()
        command = re.sub(r"[^\w\s]", "", command)
        output["orig_text"] = command

        tokens, phrase_mask = self.corpus.tokenize(output["orig_text"])
        output["text"] = tokens
        output["text_mask"] = phrase_mask

        return output
