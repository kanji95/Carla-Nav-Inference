import wandb

import re
from glob import glob

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.models._utils import IntermediateLayerGetter

from models.model import *
from dataloader.word_utils import Corpus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpu = torch.cuda.device_count()
print(f"Using {device} with {num_gpu} GPUS!")

experiment = wandb.init(project="Language Navigation", dir="/tmp")

val_path = "/ssd_scratch/cvit/kanishk/carla_data/val/"
glove_path = "/ssd_scratch/cvit/kanishk/glove"
checkpoint_path = "./saved_model/17_Feb_11-51.pth"

corpus = Corpus(glove_path)

return_layers = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}
img_backbone = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
visual_encoder = IntermediateLayerGetter(img_backbone, return_layers)

network = IROSBaseline(visual_encoder, hidden_dim=256, mask_dim=448)

if num_gpu > 1:
    network = nn.DataParallel(network)
    print("Using DataParallel mode!")
network.to(device)

checkpoint = torch.load(checkpoint_path)
network.load_state_dict(checkpoint["state_dict"])

network.eval()

img_transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        #                      0.229, 0.224, 0.225]),
    ]
)

mask_transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ]
)

frame_mask = torch.ones(1, 14 * 14, dtype=torch.int64).cuda(non_blocking=True)

columns = ["Command"]

episodes = glob(val_path + "*")
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
        gt_mask = mask_transform(gt_mask).cuda(non_blocking=True).unsqueeze(0)

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
    gt_mask_video_overlay = np.clip(gt_mask_video_overlay, a_min=0., a_max=1.)

    frame_video = np.uint8(frame_video * 255)
    mask_video = np.uint8(mask_video_overlay * 255)
    gt_mask_video = np.uint8(gt_mask_video_overlay * 255)

    print(command)

    wandb.log(
        {
            "video": wandb.Video(frame_video, fps=4, caption=command, format="gif"),
            "pred_mask": wandb.Video(mask_video, fps=4, caption=command, format="gif"),
            "gt_mask": wandb.Video(gt_mask_video, fps=4, caption=command, format="gif"),
        }
    )
