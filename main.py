import wandb

import os
import argparse
from datetime import datetime
import numpy as np

import torch

from solver import Solver

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_gpu = torch.cuda.device_count()
# print(f"Using {device} with {num_gpu} GPUS!")


def main(args):

    print(args)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_filename = os.path.join(
        save_path,
        f'{args.img_backbone}_{args.attn_type}_hd_{args.hidden_dim}_sf_{args.one_in_n}_tf_{args.traj_frames}_{datetime.now().strftime("%d_%b_%H_%M")}.pth',
    )

    print(
        f"================= Model Filename: {model_filename} =================")

    print("Initializing Solver!")
    solver = Solver(args)

    best_loss = 10e9
    epochs_without_improvement = 0
    print(
        f"Training Iterations: {len(solver.train_loader)}, Validation Iterations: {len(solver.val_loader)}")

    for epoch in range(args.epochs):
        solver.train(epoch)
        _, val_loss = solver.evaluate(epoch)

        solver.lr_scheduler.step(val_loss)

        if val_loss <= best_loss:
            best_loss = val_loss

            print(
                f"Saving Checkpoint at epoch {epoch}, best validation loss is {best_loss}!"
            )
            if args.save:
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": solver.network.state_dict(),
                        "optimizer": solver.optimizer.state_dict(),
                    },
                    model_filename,
                )
            epochs_without_improvement = 0
        elif val_loss > best_loss and epoch != args.epochs - 1:
            epochs_without_improvement += 1
            print(f"Epochs without Improvement: {epochs_without_improvement}")

            if epochs_without_improvement == 10:
                print(
                    f"{epochs_without_improvement} epochs without improvement, Stopping Training!"
                )
                break

    if args.save:
        print(f"Current Model Name {model_filename}")
        new_filename = os.path.join(
            save_path, f'{args.img_backbone}_{args.attn_type}_hd_{args.hidden_dim}_sf_{args.one_in_n}_tf_{args.traj_frames}_{best_loss:.5f}.pth')
        os.rename(model_filename, new_filename)
        print(f"Renamed to {new_filename}!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VLN Navigation")
    parser.add_argument("--seed", default=420, type=int, help="random seed")
    parser.add_argument("--epochs", default=200, type=int, help="epoch size")

    parser.add_argument("--batch_size", default=100,
                        type=int, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=10, help="number of workers")

    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--gamma", default=0.7, type=float)
    parser.add_argument(
        "--optimizer",
        default="AdamW",
        choices=["AdamW", "Adam", "SGD", "RMSprop", "Rprop", "ASGD", "RAdam"],
        type=str,
    )

    parser.add_argument(
        "--data_root",
        default="/ssd_scratch/cvit/kanishk/carla_data",
        type=str,
        help="dataset name",
    )

    parser.add_argument(
        "--glove_path",
        default="/ssd_scratch/cvit/kanishk/glove",
        type=str,
        help="dataset name",
    )

    parser.add_argument(
        "--model",
        default='baseline',
        choices=[
            'baseline'
        ],
        type=str,
    )

    parser.add_argument(
        "--imtext_matching",
        default='cross_attention',
        choices=[
            'cross_attention',
            'concat',
            'avg_concat',
        ],
        type=str,
    )
    
    parser.add_argument(
        "--attn_type",
        default='dot_product',
        choices=[
            'dot_product',
            'scaled_dot_product',
            'multi_head',
            'rel_multi_head',
            'custom_attn'
        ],
        type=str,
    )

    parser.add_argument(
        "--img_backbone",
        default="vit_tiny_patch16_224",
        choices=[
            "vit_tiny_patch16_224",
            "vit_small_patch16_224",
            "vit_tiny_patch16_384",
            "vit_small_patch16_384",
            "dino_resnet50",
            "timesformer",
            "deeplabv3_resnet50",
            "deeplabv3_resnet101",
            "deeplabv3_mobilenet_v3_large",
            "convlstm",
        ],
        type=str,
    )

    parser.add_argument(
        "--loss_func",
        default='bce',
        choices=[
            'bce', 'combo', 'class_level_bce', 'class_level_kldiv', 'class_level_combo'
        ],
        type=str,
    )

    parser.add_argument("--image_dim", type=int,
                        default=224, help="Image Dimension")
    parser.add_argument("--mask_dim", type=int,
                        default=112, help="Mask Dimension")
    parser.add_argument("--traj_dim", type=int,
                        default=56, help="Trajectory Mask Dimension")
    parser.add_argument("--hidden_dim", type=int,
                        default=256, help="Hidden Dimension")
    parser.add_argument("--num_frames", type=int,
                        default=16, help="Frames of Video")
    parser.add_argument("--traj_frames", type=int,
                        default=16, help="Next Frames of Trajectory")
    parser.add_argument("--traj_size", type=int,
                        default=25, help="Trajectory Size")
    parser.add_argument("--one_in_n", type=int,
                        default=20, help="Image Dimension")

    parser.add_argument("--patch_size", type=int,
                        default=16, help="Patch Size of Video Frame for ViT")

    parser.add_argument("--grad_check", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default="./saved_model")

    parser.add_argument("--threshold", type=float,
                        default=0.4, help="mask threshold")

    parser.add_argument("--save", default=False, action="store_true")

    args = parser.parse_args()

    main(args)
