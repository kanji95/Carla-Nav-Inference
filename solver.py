from operator import gt
import wandb

import gc
import psutil
from time import time
from datetime import datetime

import torch
from torch.optim import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models._utils import IntermediateLayerGetter

import timm

from models.model import *
from dataloader.carla_loader import *
from utilities.metrics import *
from utilities.utilities import *


class Solver(object):

    def __init__(self, args):
        self.args = args
        
        self.experiment = wandb.init(project="Language Navigation", config=self.args)

        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.lr = self.args.lr
        self.weight_decay = self.args.weight_decay
        self.gamma = self.args.gamma
        self.num_workers = self.args.num_workers
        
        self.data_root = self.args.data_root
        
        self.img_backbone = self.args.img_backbone
        self.image_dim = self.args.image_dim
        self.mask_dim = self.args.mask_dim
        self.hidden_dim = self.args.hidden_dim
        
        self.grad_check = self.args.grad_check
        
        self.threshold = self.args.threshold
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpu = torch.cuda.device_count()
        print(f"Using {self.device} with {self.num_gpu} GPUS!")
        
        return_layers = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}
        
        if "vit_" in self.img_backbone:
            img_backbone = timm.create_model(self.img_backbone, pretrained=True)
            visual_encoder = nn.Sequential(*list(img_backbone.children())[:-1])
        elif "dino_resnet50" in self.img_backbone:
            img_backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            visual_encoder = IntermediateLayerGetter(img_backbone, return_layers)
        
        self.network = IROSBaseline(visual_encoder, hidden_dim=self.hidden_dim, mask_dim=self.mask_dim)
        
        wandb.watch(self.network, log="all")
        
        self.log_parameter_info()
        
        if self.num_gpu > 1:
            self.network = nn.DataParallel(self.network)
            print("Using DataParallel mode!")
        self.network.to(self.device)
        
        self.optimizer = self.initialize_optimizer()
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=self.gamma,
            patience=2,
            threshold=1e-3,
            min_lr=1e-6,
            verbose=True,
        )
        
        train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_dim, self.image_dim)),
                transforms.RandomGrayscale(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
        val_transform = transforms.Compose(
            [
                transforms.Resize((self.image_dim, self.image_dim)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
        mask_transform = transforms.Compose(
            [
                transforms.Resize((self.mask_dim, self.mask_dim)),
                transforms.ToTensor(),
            ]
        )
        
        self.train_dataset = CarlaDataset(
            data_root=self.data_root, split="train", dataset_len=100000, 
            img_transform=train_transform, mask_transform=mask_transform
        )
        self.val_dataset = CarlaDataset(
            data_root=self.data_root, split="val", dataset_len=20000,
            img_transform=val_transform, mask_transform=mask_transform
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )
        
        self.criterion = nn.BCELoss(reduction='mean')
        

    def initialize_optimizer(self):
        params = list([p for p in self.network.parameters() if p.requires_grad])
        
        print(f"Using {self.args.optimizer} optimizer!!")
        if self.args.optimizer == "AdamW":
            optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.args.optimizer == "Adam":
            optimizer = Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.args.optimizer == "SGD":
            optimizer = SGD(params, lr=self.lr, momentum=0.8, weight_decay=self.weight_decay)
        elif self.args.optimizer == "RMSprop":
            optimizer = RMSprop(params, lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=self.weight_decay, momentum=0.8, centered=False)
        elif self.args.optimizer == "Rprop":
            optimizer = Rprop(params, lr=self.lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
        elif self.args.optimizer == "RAdam":
            optimizer = RAdam(params, lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.weight_decay)
        elif self.args.optimizer == "ASGD":
            optimizer = ASGD(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    

    def log_parameter_info(self):
        total_parameters = 0
        for name, child in self.network.named_children():
            num_params = sum([p.numel() for p in child.parameters() if p.requires_grad])
            if num_params > 0:
                print(f"No. of params in {name}: {num_params}")
                total_parameters += num_params

        print(f"Total number of params: {total_parameters}")
        
    def train(self, epochId):
        
        pid = os.getpid()
        py = psutil.Process(pid)
    
        self.network.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        total_inter, total_union = 0, 0
        total_pg = 0
        
        data_len = len(self.train_loader)

        num_samples = 0
        
        epoch_start = time()
        for step, batch in enumerate(self.train_loader):
            iterId = step + (epochId * data_len) - 1
            with torch.no_grad():
                frame = batch["frame"].cuda(non_blocking=True)
                text = batch["text"].cuda(non_blocking=True)
                text_mask = batch["text_mask"].cuda(non_blocking=True)
                gt_mask = batch["gt_frame"].cuda(non_blocking=True)
                
                batch_size = frame.shape[0]
                num_samples += batch_size
            
            start_time = time()
            
            mask = self.network(frame, text, text_mask)
            
            loss = self.criterion(mask, gt_mask)
            loss.backward()
            
            if iterId % 1000 == 0 and self.grad_check:
                grad_check(self.network.named_parameters())
            
            self.optimizer.step()

            self.network.zero_grad()

            end_time = time()
            elapsed_time = end_time - start_time
            
            with torch.no_grad():
                inter, union = compute_mask_IOU(mask, gt_mask, self.threshold)

            total_inter += inter.item()
            total_union += union.item()
            
            total_pg += pointing_game(mask, gt_mask)

            total_loss += float(loss.item())
            
            if step % 500 == 0:
                log_predicitons(
                    batch["orig_frame"],
                    batch["orig_text"],
                    mask.detach().cpu(),
                    gt_mask.detach().cpu(),
                    title="training",
                )

            if iterId % 100 == 0 and step != 0:
            # import pdb; pdb.set_trace()
                print(mask.min(), mask.max())
                gc.collect()
                memoryUse = py.memory_info()[0] / 2.0 ** 20
                timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
                curr_loss = total_loss / (step + 1)
                curr_IOU = total_inter / total_union
                curr_pg = total_pg / num_samples
                lr = self.optimizer.param_groups[0]["lr"]
                
                print(
                        f"{timestamp} Epoch:[{epochId:2d}/{self.epochs:2d}] iter {iterId:6d} loss {curr_loss:.4f} IOU {curr_IOU:.4f} PG {curr_pg:.4f} memory_use {memoryUse:.3f}MB lr {lr:.7f} elapsed {elapsed_time:.2f}"
                )
                
        epoch_end = time()
        epoch_time = epoch_end - epoch_start

        timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

        train_loss = total_loss / data_len
        train_IOU = total_inter / total_union
        train_pg = total_pg / num_samples
        
        wandb.log(
            {
                "loss": train_loss,
                "IOU": train_IOU,
                "PG": train_pg,
            }
        )
        
        print(
                f"{timestamp} FINISHED Epoch:{epochId:2d} loss {train_loss:.4f} overall_IOU {train_IOU:.4f} PG {train_pg:.4f} elapsed {epoch_time:.2f}"
        )
    
    @torch.no_grad()
    def evaluate(self, epochId):

        pid = os.getpid()
        py = psutil.Process(pid)
        
        self.network.eval()

        total_loss = 0
        total_inter, total_union = 0, 0
        total_pg = 0
        
        data_len = len(self.val_loader)

        num_samples = 0
        
        for step, batch in enumerate(self.val_loader):
            frame = batch["frame"].cuda(non_blocking=True)
            text = batch["text"].cuda(non_blocking=True)
            text_mask = batch["text_mask"].cuda(non_blocking=True)
            gt_mask = batch["gt_frame"].cuda(non_blocking=True)
            
            batch_size = frame.shape[0]
            num_samples += batch_size

            start_time = time()
            
            mask = self.network(frame, text, text_mask)
            loss = self.criterion(mask, gt_mask)
            
            end_time = time()
            elapsed_time = end_time - start_time
            
            inter, union = compute_mask_IOU(mask, gt_mask, self.threshold)

            total_inter += inter.item()
            total_union += union.item()
            
            total_pg += pointing_game(mask, gt_mask)

            total_loss += float(loss.item())

            if step % 500 == 0:
                log_predicitons(
                    batch["orig_frame"],
                    batch["orig_text"],
                    mask.detach().cpu(),
                    gt_mask.detach().cpu(),
                    title="validation",
                )
            if step % 50 == 0:
                print(mask.min(), mask.max())
                
                gc.collect()
                memoryUse = py.memory_info()[0] / 2.0 ** 20

                timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

                curr_loss = total_loss / (step + 1)
                curr_IOU = total_inter / total_union
                curr_pg = total_pg / num_samples
                
                print(
                    f"{timestamp} Validation: iter [{step:3d}/{data_len}] loss {curr_loss:.4f} overall_IOU {curr_IOU:.4f} pointing_game {curr_pg:.4f} memory_use {memoryUse:.3f}MB elapsed {elapsed_time:.2f}"
                )
        
        val_loss = total_loss / data_len
        val_IOU = total_inter / total_union
        val_pg = total_pg / num_samples
        
        timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
        
        wandb.log(
            {
                "val_loss": val_loss,
                "val_IOU": val_IOU,
                "val_PG": val_pg,
            }
        )
        
        print(
                f"{timestamp} Validation: EpochId: {epochId:2d} loss {val_loss:.4f} overall_IOU {val_IOU:.4f} pointing_game {val_pg:.4f}"
        )
        
        return val_IOU, val_loss
