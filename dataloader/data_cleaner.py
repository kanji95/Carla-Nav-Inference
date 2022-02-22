import os
from glob import glob

import numpy as np

from torch.utils.data import Dataset

from word_utils import Corpus


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
        dataset_len=10000,
        skip=5,
        sequence_len=16,
        mode="image",
        image_dim=224,
        mask_dim=112,
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
            self.dataset_len = self.dataset_len // self.sequence_len

        self.episodes = sorted(os.listdir(self.data_dir))
        # print(self.episodes)

        self.corpus = Corpus(glove_path)

    def get_image_data(
        self, K, image_files, mask_files, matrix_files, vehicle_positions, T=10
    ):

        num_files = len(image_files)
        # print("Number of files: ", num_files)
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
            # print("====================")
            # print(pixel_t_2d)
            
            if (0 < pixel_t_2d[0] < 1280) and (0 < pixel_t_2d[1] < 720):
                break
            
            # print("before sample idx: ", sample_idx)
            sample_idx += 1
            sample_idx %= num_files - T
            # print# ("after sample idx: ", sample_idx)
            # print("====================")
            if prev_idx == sample_idx:
                return False

        return True

        # train -> 109 113 114 121 128 131 132 140 146 15 152 155 156 159 161 166 171 172 177 179 19 195 206 214 215 216 222 230 27 30 31 34 35 49 58 59 61 68 7 72 73 74 81 82 83 86 88 91 92 96 98 54
        # val -> 1 11 14 18 2 25 28 32 33 34 37 39 44 46 5 50 7 

    def get_data(self, episode):
        episode_dir = os.path.join(self.data_dir, episode)

        image_files = sorted(glob(episode_dir + f"/images/*.png"))
        mask_files = sorted(glob(episode_dir + f"/masks/*.png"))
        matrix_files = sorted(glob(episode_dir + f"/inverse_matrix/*.npy"))
        position_file = os.path.join(episode_dir, "vehicle_positions.txt")
        intrinsic_path = os.path.join(episode_dir, "camera_intrinsic.npy")

        K = np.load(intrinsic_path)

        vehicle_positions = []
        with open(position_file, "r") as fhand:
            for line in fhand:
                position = np.array(line.split(","), dtype=np.float32)
                vehicle_positions.append(position)

        if (
            self.get_image_data(
                K, image_files, mask_files, matrix_files, vehicle_positions
            )
            is False
        ):
            print(episode, end=" ")
            return episode
        return ''


if __name__ == "__main__":
    dataset = CarlaFullDataset(
        "/ssd_scratch/cvit/kanishk/carla_data",
        "/ssd_scratch/cvit/kanishk/glove",
        "train",
    )
    ignore_episodes = []
    for episode in dataset.episodes:
        ret = dataset.get_data(episode)
        if ret.isnumeric():
            ignore_episodes.append(ret)
    print()
    print(ignore_episodes)
