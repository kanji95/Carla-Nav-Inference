import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from word_utils import Corpus


class CarlaDataset(Dataset):
    """Some Information about CarlaDataset"""

    def __init__(
        self,
        args,
        data_root="/ssd_scratch/cvit/kanishk/carla_data",
        split="train",
        glove_path="",
        transform=None,
        dataset_len=10000,
        skip=10,
    ):
        self.data_dir = os.path.join(data_root, split)
        self.transform = transform
        self.dataset_len = dataset_len
        self.skip = skip
        self.episodes = sorted(os.listdir(self.data_dir))
        self.corpus = Corpus(glove_path)
        corpus_path = os.path.join(self.data_dir, "corpus.pth")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        output = {}

        episode_dir = os.path.join(self.data_dir, np.random.choice(self.episodes)[0])

        image_files = sorted(glob(episode_dir + f"/images/*.png"))
        matrix_files = sorted(glob(episode_dir + f"/inverse_matrix/*.npy"))
        position_file = os.path.join(episode_dir, "vehicle_positions.txt")
        command_path = os.path.join(episode_dir, "command.txt")

        num_files = len(image_files)

        vehicle_positions = []
        with open(position_file, "r") as fhand:
            for line in fhand:
                position = np.array(line.split(","), dtype=np.float32)
                vehicle_positions.append(position)

        sample_idx = np.random.choice(range(self.skip, num_files - self.skip))

        img_path = image_files[sample_idx]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        output["image"] = np.array(img)
        output["command"] = open(command_path, "r").read()
        output["vehicle_position"] = vehicle_positions[sample_idx]
        output["matrix"] = np.load(matrix_files[sample_idx])
        output["next_vehicle_position"] = vehicle_positions[sample_idx + 1]

        tokens, phrase_mask = self.corpus.tokenize(output["command"])
        output["tokens"] = tokens
        output["phrase_mask"] = phrase_mask

        return output
