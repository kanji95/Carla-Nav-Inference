import os
import re
from glob import glob
from collections import Iterable

import numpy as np
import torch
from PIL import Image

from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset

from .word_utils import Corpus

class ResizeAnnotation:
    """Resize the largest of the sides of the annotation to a given size"""

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale_h, scale_w = self.size / im_h, self.size / im_w
        resized_h = int(np.round(im_h * scale_h))
        resized_w = int(np.round(im_w * scale_w))
        out = (
            F.interpolate(
                Variable(img).unsqueeze(0).unsqueeze(0),
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .data
        )
        return out

class CarlaDataset(Dataset):
    """Some Information about CarlaDataset"""

    def __init__(
        self,
        data_root="/ssd_scratch/cvit/kanishk/carla_data",
        split="train",
        glove_path="/ssd_scratch/cvit/kanishk/glove",
        img_transform=None,
        mask_transform=None,
        dataset_len=10000,
        skip=10,
    ):
        self.data_dir = os.path.join(data_root, split)
        
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
        self.dataset_len = dataset_len
        self.skip = skip
        self.episodes = sorted(os.listdir(self.data_dir))
        #  print(self.episodes)
        self.corpus = Corpus(glove_path)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        output = {}

        # import pdb; pdb.set_trace()
        episode_dir = os.path.join(self.data_dir, np.random.choice(self.episodes))
        # print(episode_dir)

        image_files = sorted(glob(episode_dir + f"/images/*.png"))
        mask_files = sorted(glob(episode_dir + f"/masks/*.png"))
        matrix_files = sorted(glob(episode_dir + f"/inverse_matrix/*.npy"))
        position_file = os.path.join(episode_dir, "vehicle_positions.txt")
        command_path = os.path.join(episode_dir, "command.txt")

        num_files = len(image_files)

        vehicle_positions = []
        with open(position_file, "r") as fhand:
            for line in fhand:
                position = np.array(line.split(","), dtype=np.float32)
                vehicle_positions.append(position)

        # print(num_files, episode_dir)
        sample_idx = np.random.choice(range(self.skip, num_files - self.skip))

        img_path = image_files[sample_idx]
        mask_path = mask_files[sample_idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')

        output["orig_frame"] = np.array(img)
        
        if self.img_transform:
            img = self.img_transform(img)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask[mask > 0] = 1

        output["frame"] = img
        output["gt_frame"] = mask
        
        command = open(command_path, "r").read()
        command = re.sub(r'[^\w\s]','',command)
        output["orig_text"] = command
        # print(output["orig_text"])
        # output["vehicle_position"] = vehicle_positions[sample_idx]
        # output["matrix"] = np.load(matrix_files[sample_idx])
        # output["next_vehicle_position"] = vehicle_positions[sample_idx + 1]

        tokens, phrase_mask = self.corpus.tokenize(output["orig_text"])
        output["text"] = tokens
        output["text_mask"] = phrase_mask

        return output
