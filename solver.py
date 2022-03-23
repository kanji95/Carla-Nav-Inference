# from operator import gt
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
from timesformer.models.vit import VisionTransformer

from segmentation_models_pytorch.losses import *

from models.model import *
from dataloader.carla_loader import *
from utilities.loss import *
from utilities.metrics import *
from utilities.utilities import *


class Solver(object):
    def __init__(self, args):
        self.args = args

        self.experiment = wandb.init(project="Language Navigation", config=self.args)
        self.experiment.name = f"{args.img_backbone}_{args.loss_func}_{args.attn_type}_hd_{args.hidden_dim}_sf_{args.one_in_n}_tf_{args.traj_frames}_{self.experiment.id}"

        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.lr = self.args.lr
        self.weight_decay = self.args.weight_decay
        self.gamma = self.args.gamma
        self.num_workers = self.args.num_workers

        self.data_root = self.args.data_root
        self.glove_path = self.args.glove_path

        self.loss_func = self.args.loss_func

        self.img_backbone = self.args.img_backbone
        self.imtext_matching = self.args.imtext_matching
        self.attn_type = self.args.attn_type

        self.image_dim = self.args.image_dim
        self.mask_dim = self.args.mask_dim
        self.traj_dim = self.args.traj_dim
        self.hidden_dim = self.args.hidden_dim
        self.num_frames = self.args.num_frames
        self.traj_frames = self.args.traj_frames
        self.traj_size = self.args.traj_size
        self.patch_size = self.args.patch_size
        self.one_in_n = self.args.one_in_n

        self.grad_check = self.args.grad_check

        self.threshold = self.args.threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpu = torch.cuda.device_count()
        print(f"Using {self.device} with {self.num_gpu} GPUS!")

        return_layers = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}

        self.mode = "image"
        if "vit_" in self.img_backbone:
            img_backbone = timm.create_model(self.img_backbone, pretrained=True)
            visual_encoder = nn.Sequential(*list(img_backbone.children())[:-1])
            self.network = JointSegmentationBaseline(
                visual_encoder,
                hidden_dim=self.hidden_dim,
                image_dim=self.image_dim,
                mask_dim=self.mask_dim,
                traj_dim=self.traj_dim,
                backbone=self.img_backbone,
            )
        elif "dino_resnet50" in self.img_backbone:
            img_backbone = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
            visual_encoder = IntermediateLayerGetter(img_backbone, return_layers)
            self.network = IROSBaseline(
                visual_encoder,
                hidden_dim=self.hidden_dim,
                image_dim=self.image_dim,
                mask_dim=self.mask_dim,
            )
        elif "timesformer" in self.img_backbone:
            self.mode = "video"
            spatial_dim = self.image_dim // self.patch_size
            visual_encoder = VisionTransformer(
                img_size=self.image_dim,
                patch_size=self.patch_size,
                embed_dim=self.hidden_dim,
                depth=2,
                num_heads=8,
                num_frames=self.num_frames,
            )
            self.network = JointVideoSegmentationBaseline(
                visual_encoder,
                hidden_dim=self.hidden_dim,
                image_dim=self.image_dim,
                mask_dim=self.mask_dim,
                traj_dim=self.traj_dim,
                spatial_dim=spatial_dim,
                num_frames=self.num_frames,
            )
        elif "deeplabv3_" in self.img_backbone:
            img_backbone = torch.hub.load(
                "pytorch/vision:v0.10.0", self.img_backbone, pretrained=True
            )
            visual_encoder = nn.Sequential(
                *list(img_backbone._modules["backbone"].children())
            )
            self.network = JointSegmentationBaseline(
                visual_encoder,
                hidden_dim=self.hidden_dim,
                image_dim=self.image_dim,
                mask_dim=self.mask_dim,
                traj_dim=self.traj_dim,
                backbone=self.img_backbone,
            )
        elif "convlstm" in self.img_backbone:
            self.mode = "video"
            spatial_dim = self.image_dim // self.patch_size

            video_encoder = torch.hub.load(
                "facebookresearch/pytorchvideo", "x3d_s", pretrained=True
            )
            visual_encoder = nn.Sequential(*list(video_encoder.blocks.children())[:-1])

            self.network = ConvLSTMBaseline(
                visual_encoder,
                hidden_dim=self.hidden_dim,
                image_dim=self.image_dim,
                mask_dim=self.mask_dim,
                traj_dim=self.traj_dim,
                spatial_dim=spatial_dim,
                num_frames=self.num_frames,
                attn_type=self.attn_type,
            )

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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225]),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize((self.image_dim, self.image_dim)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225]),
            ]
        )

        mask_transform = transforms.Compose(
            [
                transforms.Resize((self.mask_dim, self.mask_dim)),
                transforms.ToTensor(),
            ]
        )

        traj_transform = transforms.Compose(
            [
                transforms.Resize((self.traj_dim, self.traj_dim)),
                transforms.ToTensor(),
            ]
        )

        self.train_dataset = CarlaFullDataset(
            data_root=self.data_root,
            glove_path=self.glove_path,
            split="train",
            dataset_len=100000,
            img_transform=train_transform,
            mask_transform=mask_transform,
            traj_transform=traj_transform,
            sequence_len=self.num_frames,
            mode=self.mode,
            image_dim=self.image_dim,
            mask_dim=self.mask_dim,
            traj_dim=self.traj_dim,
            traj_frames=self.traj_frames,
            traj_size=self.traj_size,
            one_in_n=self.one_in_n,
        )
        self.val_dataset = CarlaFullDataset(
            data_root=self.data_root,
            glove_path=self.glove_path,
            split="val",
            dataset_len=20000,
            img_transform=val_transform,
            mask_transform=mask_transform,
            traj_transform=traj_transform,
            sequence_len=self.num_frames,
            mode=self.mode,
            image_dim=self.image_dim,
            mask_dim=self.mask_dim,
            traj_dim=self.traj_dim,
            traj_frames=self.traj_frames,
            traj_size=self.traj_size,
            one_in_n=self.one_in_n,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        self.bce_loss = nn.BCELoss(reduction="mean")
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.combo_loss = ComboLoss(alpha=0.8, ce_ratio=0.4)
        self.class_level_loss = ClassLevelLoss(self.loss_func, beta=0.6)

        self.focal_loss = FocalLoss(mode="binary", alpha=0.5, ignore_index=0)
        self.tversky_loss = TverskyLoss(
            mode="binary", alpha=0.5, beta=0.5, ignore_index=0
        )
        self.lovasz_loss = LovaszLoss(mode="binary", ignore_index=0)

    def initialize_optimizer(self):
        params = list([p for p in self.network.parameters() if p.requires_grad])

        print(f"Using {self.args.optimizer} optimizer!!")
        if self.args.optimizer == "AdamW":
            optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.args.optimizer == "Adam":
            optimizer = Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.args.optimizer == "SGD":
            optimizer = SGD(
                params, lr=self.lr, momentum=0.8, weight_decay=self.weight_decay
            )
        elif self.args.optimizer == "RMSprop":
            optimizer = RMSprop(
                params,
                lr=self.lr,
                alpha=0.99,
                eps=1e-08,
                weight_decay=self.weight_decay,
                momentum=0.8,
                centered=False,
            )
        elif self.args.optimizer == "Rprop":
            optimizer = Rprop(
                params, lr=self.lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50)
            )
        # elif self.args.optimizer == "RAdam":
        #     optimizer = RAdam(
        #         params,
        #         lr=self.lr,
        #         betas=(0.9, 0.999),
        #         eps=1e-08,
        #         weight_decay=self.weight_decay,
        #     )
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
        total_inter_mask, total_union_mask = 0, 0
        total_inter_traj, total_union_traj = 0, 0
        total_pg_mask, total_pg_traj = 0, 0
        total_it_mask, total_it_traj = 0, 0
        total_rk_mask, total_rk_traj = 0, 0
        total_ds_mask, total_ds_traj = 0, 0

        data_len = len(self.train_loader)

        num_samples = 0

        epoch_start = time()
        for step, batch in enumerate(self.train_loader):
            iterId = step + (epochId * data_len) - 1
            with torch.no_grad():
                frame = batch["frame"].cuda(non_blocking=True)

                text = batch["text"].cuda(non_blocking=True)
                sub_text = batch["sub_text"].cuda(non_blocking=True)

                text_mask = batch["text_mask"].cuda(non_blocking=True)
                sub_text_mask = batch["sub_text_mask"].cuda(non_blocking=True)
                
                sub_text_labels = batch["sub_text_labels"].cuda(non_blocking=True)

                gt_mask = batch["gt_frame"].cuda(non_blocking=True)
                gt_traj_mask = batch["gt_traj_mask"].cuda(non_blocking=True)

                batch_size = frame.shape[0]
                frame_mask = torch.ones(batch_size, 7 * 7, dtype=torch.int64).cuda(
                    non_blocking=True
                )
                num_samples += batch_size

                # re_mask = rearrange(mask, "b c t h w -> (b t) c h w")
                re_gt_mask = rearrange(gt_mask, "b c t h w -> (b t) c h w")
                bs, _, h, w = re_gt_mask.shape

                new_gt_mask = torch.zeros(bs, h, w).cuda(non_blocking=True)
                new_gt_mask[re_gt_mask[:, 0] == 1] = 1
                new_gt_mask[re_gt_mask[:, 1] == 1] = 2

            start_time = time()

            # import pdb; pdb.set_trace()
            mask, traj_mask, sub_cmd_wts = self.network(
                frame, text, sub_text, frame_mask, text_mask, sub_text_mask
            )
            re_mask = rearrange(mask, "b c t h w -> (b t) c h w")

            if self.loss_func == "bce":
                loss = self.bce_loss(re_mask, new_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            elif self.loss_func == "combo":
                loss = self.combo_loss(re_mask, new_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            elif "class_level" in self.loss_func:
                loss = self.class_level_loss(re_mask, re_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            elif "focal" in self.loss_func:
                loss = self.focal_loss(re_mask, re_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            elif "tversky" in self.loss_func:
                loss = self.tversky_loss(re_mask, re_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            elif "lovasz" in self.loss_func:
                loss = self.lovasz_loss(re_mask, re_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            else:
                raise NotImplementedError(f"{self.loss_func} not implemented!")

            loss.backward()

            if iterId % 1000 == 0 and self.grad_check:
                grad_check(self.network.named_parameters())

            self.optimizer.step()

            self.network.zero_grad()

            end_time = time()
            elapsed_time = end_time - start_time

            with torch.no_grad():
                inter_mask, union_mask = compute_mask_IOU(mask, gt_mask, self.threshold)
                inter_traj, union_traj = compute_mask_IOU(
                    traj_mask, gt_traj_mask, self.threshold
                )

            total_inter_mask += inter_mask.item()
            total_union_mask += union_mask.item()

            total_inter_traj += inter_traj.item()
            total_union_traj += union_traj.item()

            total_pg_mask += pointing_game(mask, gt_mask)
            total_pg_traj += pointing_game(traj_mask, gt_traj_mask)

            total_it_mask += intersection_at_t(mask, gt_mask)
            total_it_traj += intersection_at_t(traj_mask, gt_traj_mask)

            total_rk_mask += recall_at_k(mask, gt_mask)
            total_rk_traj += recall_at_k(traj_mask, gt_traj_mask)

            total_ds_mask += dice_score(mask, gt_mask)
            total_ds_traj += dice_score(traj_mask, gt_traj_mask)

            total_loss += float(loss.item())

            if step % 1000 == 0:
                if self.mode == "image":
                    log_frame_predicitons(
                        batch["orig_frame"],
                        batch["orig_text"],
                        mask.detach().cpu(),
                        traj_mask.detach().cpu(),
                        gt_mask.detach().cpu(),
                        gt_traj_mask.detach().cpu(),
                        batch["episode"],
                        batch["sample_idx"],
                        title="training",
                    )
                else:
                    log_video_predicitons(
                        batch["orig_frame"],
                        batch["orig_text"],
                        mask.detach().cpu(),
                        traj_mask.detach().cpu(),
                        gt_mask.detach().cpu(),
                        gt_traj_mask.detach().cpu(),
                        batch["episode"],
                        batch["sample_idx"],
                        title="training",
                    )

            if iterId % 250 == 0 and step != 0:
                # import pdb; pdb.set_trace()
                # print(mask.min(), mask.max())
                gc.collect()
                memoryUse = py.memory_info()[0] / 2.0 ** 20
                timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
                curr_loss = total_loss / (step + 1)

                curr_IOU_mask = total_inter_mask / total_union_mask
                curr_IOU_traj = total_inter_traj / total_union_traj

                curr_pg_mask = total_pg_mask / num_samples
                curr_pg_traj = total_pg_traj / num_samples

                curr_it_mask = total_it_mask / num_samples
                curr_it_traj = total_it_traj / num_samples

                curr_rk_mask = total_rk_mask / num_samples
                curr_rk_traj = total_rk_traj / num_samples

                curr_ds_mask = total_ds_mask / num_samples
                curr_ds_traj = total_ds_traj / num_samples

                lr = self.optimizer.param_groups[0]["lr"]

                print(
                    f"{timestamp} Epoch:[{epochId:2d}/{self.epochs:2d}] iter {iterId:6d} loss {curr_loss:.4f} |Mask IOU {curr_IOU_mask:.4f}|Traj IOU {curr_IOU_traj:.4f}|Mask PG {curr_pg_mask:.4f}|Traj PG {curr_pg_traj:.4f}|Mask IT {curr_it_mask:.4f}|Traj IT {curr_it_traj:.4f}|Mask RK {curr_rk_mask:.4f}|Traj RK {curr_rk_traj:.4f}|Mask DS {curr_ds_mask:.4f}|Traj DS {curr_ds_traj:.4f}| mem_use {memoryUse:.3f}MB elapsed {elapsed_time:.2f}"
                )

                # print(
                #     f"{timestamp} Epoch:[{epochId:2d}/{self.epochs:2d}] iter {iterId:6d} loss {curr_loss:.4f} Mask IOU {curr_IOU_mask:.4f} Traj IOU {curr_IOU_traj:.4f} Mask PG {curr_pg_mask:.4f} Traj PG {curr_pg_traj:.4f} memory_use {memoryUse:.3f}MB lr {lr:.7f} elapsed {elapsed_time:.2f}"
                # )

        epoch_end = time()
        epoch_time = epoch_end - epoch_start

        timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

        train_loss = total_loss / data_len

        train_IOU_mask = total_inter_mask / total_union_mask
        train_IOU_traj = total_inter_traj / total_union_traj

        train_pg_mask = total_pg_mask / num_samples
        train_pg_traj = total_pg_traj / num_samples

        train_it_mask = total_it_mask / num_samples
        train_it_traj = total_it_traj / num_samples

        train_rk_mask = total_rk_mask / num_samples
        train_rk_traj = total_rk_traj / num_samples

        train_ds_mask = total_ds_mask / num_samples
        train_ds_traj = total_ds_traj / num_samples

        wandb.log(
            {
                "loss": train_loss,
                "Mask IOU": train_IOU_mask,
                "Traj IOU": train_IOU_traj,
                "Mask PG": train_pg_mask,
                "Traj PG": train_pg_traj,
                "Mask IT": train_it_mask,
                "Traj IT": train_it_traj,
                "Mask RK": train_rk_mask,
                "Traj RK": train_rk_traj,
                "Mask DS": train_ds_mask,
                "Traj DS": train_ds_traj,
            }
        )

        print(
            f"{timestamp} FINISHED Epoch:{epochId:2d} loss {train_loss:.4f} Mask IOU {train_IOU_mask:.4f} Traj IOU {train_IOU_traj:.4f} Mask PG {train_pg_mask:.4f} Traj PG {train_pg_traj:.4f} Mask IT {train_it_mask:.4f} Traj IT {train_it_traj:.4f} Mask RK {train_rk_mask:.4f} Traj RK {train_rk_traj:.4f} Mask DS {train_ds_mask:.4f} Traj DS {train_ds_traj:.4f} elapsed {epoch_time:.2f}"
        )

    @torch.no_grad()
    def evaluate(self, epochId):

        pid = os.getpid()
        py = psutil.Process(pid)

        self.network.eval()

        total_loss = 0
        total_inter_mask, total_union_mask = 0, 0
        total_inter_traj, total_union_traj = 0, 0
        total_pg_mask, total_pg_traj = 0, 0
        total_it_mask, total_it_traj = 0, 0
        total_rk_mask, total_rk_traj = 0, 0
        total_ds_mask, total_ds_traj = 0, 0

        data_len = len(self.val_loader)

        num_samples = 0

        for step, batch in enumerate(self.val_loader):
            frame = batch["frame"].cuda(non_blocking=True)

            text = batch["text"].cuda(non_blocking=True)
            sub_text = batch["sub_text"].cuda(non_blocking=True)

            text_mask = batch["text_mask"].cuda(non_blocking=True)
            sub_text_mask = batch["sub_text_mask"].cuda(non_blocking=True)
            
            sub_text_labels = batch["sub_text_labels"].cuda(non_blocking=True)

            gt_mask = batch["gt_frame"].cuda(non_blocking=True)
            gt_traj_mask = batch["gt_traj_mask"].cuda(non_blocking=True)

            batch_size = frame.shape[0]
            frame_mask = torch.ones(batch_size, 7 * 7, dtype=torch.int64).cuda(
                non_blocking=True
            )
            num_samples += batch_size

            # re_mask = rearrange(mask, "b c t h w -> (b t) c h w")
            re_gt_mask = rearrange(gt_mask, "b c t h w -> (b t) c h w")
            bs, _, h, w = re_gt_mask.shape

            new_gt_mask = torch.zeros(bs, h, w).cuda(non_blocking=True)
            new_gt_mask[re_gt_mask[:, 0] == 1] = 1
            new_gt_mask[re_gt_mask[:, 1] == 1] = 2

            start_time = time()

            mask, traj_mask, sub_cmd_wts = self.network(
                frame, text, sub_text, frame_mask, text_mask, sub_text_mask
            )
            re_mask = rearrange(mask, "b c t h w -> (b t) c h w")

            if self.loss_func == "bce":
                loss = self.bce_loss(re_mask, new_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            elif self.loss_func == "combo":
                loss = self.combo_loss(re_mask, new_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            elif "class_level" in self.loss_func:
                loss = self.class_level_loss(re_mask, re_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            elif "focal" in self.loss_func:
                loss = self.focal_loss(re_mask, re_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            elif "tversky" in self.loss_func:
                loss = self.tversky_loss(re_mask, re_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            elif "lovasz" in self.loss_func:
                loss = self.lovasz_loss(re_mask, re_gt_mask) + self.combo_loss(
                    traj_mask, gt_traj_mask
                ) + self.bce_loss(sub_cmd_wts, sub_text_labels)
            else:
                raise NotImplementedError(f"{self.loss_func} not implemented!")

            end_time = time()
            elapsed_time = end_time - start_time

            inter_mask, union_mask = compute_mask_IOU(mask, gt_mask, self.threshold)
            inter_traj, union_traj = compute_mask_IOU(
                traj_mask, gt_traj_mask, self.threshold
            )

            total_inter_mask += inter_mask.item()
            total_union_mask += union_mask.item()

            total_inter_traj += inter_traj.item()
            total_union_traj += union_traj.item()

            total_pg_mask += pointing_game(mask, gt_mask)
            total_pg_traj += pointing_game(traj_mask, gt_traj_mask)

            total_it_mask += intersection_at_t(mask, gt_mask)
            total_it_traj += intersection_at_t(traj_mask, gt_traj_mask)

            total_rk_mask += recall_at_k(mask, gt_mask)
            total_rk_traj += recall_at_k(traj_mask, gt_traj_mask)

            total_ds_mask += dice_score(mask, gt_mask)
            total_ds_traj += dice_score(traj_mask, gt_traj_mask)

            total_loss += float(loss.item())

            if step % 500 == 0:
                if self.mode == "image":
                    log_frame_predicitons(
                        batch["orig_frame"],
                        batch["orig_text"],
                        mask.detach().cpu(),
                        traj_mask.detach().cpu(),
                        gt_mask.detach().cpu(),
                        gt_traj_mask.detach().cpu(),
                        batch["episode"],
                        batch["sample_idx"],
                        title="validation",
                    )
                else:
                    log_video_predicitons(
                        batch["orig_frame"],
                        batch["orig_text"],
                        mask.detach().cpu(),
                        traj_mask.detach().cpu(),
                        gt_mask.detach().cpu(),
                        gt_traj_mask.detach().cpu(),
                        batch["episode"],
                        batch["sample_idx"],
                        title="validation",
                    )
            if step % 250 == 0:
                # print(mask.min(), mask.max())

                gc.collect()
                memoryUse = py.memory_info()[0] / 2.0 ** 20

                timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

                curr_loss = total_loss / (step + 1)

                curr_IOU_mask = total_inter_mask / total_union_mask
                curr_IOU_traj = total_inter_traj / total_union_traj

                curr_pg_mask = total_pg_mask / num_samples
                curr_pg_traj = total_pg_traj / num_samples

                curr_it_mask = total_it_mask / num_samples
                curr_it_traj = total_it_traj / num_samples

                curr_rk_mask = total_rk_mask / num_samples
                curr_rk_traj = total_rk_traj / num_samples

                curr_ds_mask = total_ds_mask / num_samples
                curr_ds_traj = total_ds_traj / num_samples

                print(
                    f"{timestamp} Validation: iter [{step:3d}/{data_len}] loss {curr_loss:.4f} |Mask IOU {curr_IOU_mask:.4f}|Traj IOU {curr_IOU_traj:.4f}|Mask PG {curr_pg_mask:.4f}|Traj PG {curr_pg_traj:.4f}|Mask IT {curr_it_mask:.4f}|Traj IT {curr_it_traj:.4f}|Mask RK {curr_rk_mask:.4f}|Traj RK {curr_rk_traj:.4f}|Mask DS {curr_ds_mask:.4f}|Traj DS {curr_ds_traj:.4f}| mem_use {memoryUse:.3f}MB elapsed {elapsed_time:.2f}"
                )

                # print(
                #     f"{timestamp} Validation: iter [{step:3d}/{data_len}] loss {curr_loss:.4f} Mask IOU {curr_IOU_mask:.4f} Traj IOU {curr_IOU_traj:.4f} Mask PG {curr_pg_mask:.4f} Traj PG {curr_pg_traj:.4f} memory_use {memoryUse:.3f}MB elapsed {elapsed_time:.2f}"
                # )

        # print(mask.min(), mask.max())

        val_loss = total_loss / data_len

        val_IOU_mask = total_inter_mask / total_union_mask
        val_IOU_traj = total_inter_traj / total_union_traj

        val_pg_mask = total_pg_mask / num_samples
        val_pg_traj = total_pg_traj / num_samples

        val_it_mask = total_it_mask / num_samples
        val_it_traj = total_it_traj / num_samples

        val_rk_mask = total_rk_mask / num_samples
        val_rk_traj = total_rk_traj / num_samples

        val_ds_mask = total_ds_mask / num_samples
        val_ds_traj = total_ds_traj / num_samples

        timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

        wandb.log(
            {
                "val_loss": val_loss,
                "val_Mask_IOU": val_IOU_mask,
                "val_Traj_IOU": val_IOU_traj,
                "val_Mask_PG": val_pg_mask,
                "val_Traj_PG": val_pg_traj,
                "val_Mask_IT": val_it_mask,
                "val_Traj_IT": val_it_traj,
                "val_Mask_RK": val_rk_mask,
                "val_Traj_RK": val_rk_traj,
                "val_Mask_DS": val_ds_mask,
                "val_Traj_DS": val_ds_traj,
            }
        )

        print(
            f"{timestamp} Validation: EpochId: {epochId:2d} loss {val_loss:.4f} Mask_IOU {val_IOU_mask:.4f} Traj_IOU {val_IOU_traj:.4f} Mask_PG {val_pg_mask:.4f} Traj_PG {val_pg_traj:.4f} Mask IT {val_it_mask:.4f} Traj IT {val_it_traj:.4f} Mask RK {val_rk_mask:.4f} Traj RK {val_rk_traj:.4f} Mask DS {val_ds_mask:.4f} Traj DS {val_ds_traj:.4f}"
        )

        return val_pg_mask, val_loss
