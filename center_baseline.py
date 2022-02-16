import os
from glob import glob
import numpy as np
from PIL import Image


root_dir = "/ssd_scratch/cvit/kanishk/carla_data/"
# mask_dir = os.path.join(root_dir, "val_masks_new")

episodes = glob(root_dir+"val/*")

correct = 0
total = 0

for episode in episodes:
    
    mask_files = sorted(glob(episode+"/masks/*.png"))[10:, :-10]
    print(episode, len(mask_files))
    
    for mask_file in mask_files:
        mask_img = Image.open(mask_file).convert('L')
        mask_img = np.array(mask_img)

        h, w = mask_img.shape

        total += 1
        if mask_img[w//2, h//2] > 0:
            correct += 1

print(total, correct)
print(correct/total)