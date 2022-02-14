import os
import numpy as np
from glob import glob
from torch.utils.data import Dataset
        
class CarlaDataset(Dataset):
    """Some Information about CarlaDataset"""
    def __init__(self, args, data_root="/ssd_scratch/cvit/kanishk/carla_data", split="train"):
        super(CarlaDataset, self).__init__()
        
        self.split = split
        
        self.data_root = data_root
        self.img_dir = os.path.join(self.data_dir, "images")
        self.mask_dir = os.path.join(self.data_dir, "annotations")
        self.trans_dir = os.path.join(self.data_dir, "inverse_matrix")
        
        

    def __getitem__(self, index):
        return 

    def __len__(self):
        return 