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
        f'{datetime.now().strftime("%d_%b_%H-%M")}.pth',
    )

    print("Initializing Solver!")
    solver = Solver(args)

    best_IOU = 0
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        solver.train(epoch)
        val_IOU, val_loss = solver.evaluate(epoch)

        solver.lr_scheduler.step(val_loss)

        if val_IOU > best_IOU:
            best_IOU = val_IOU

            print(
                f"Saving Checkpoint at epoch {epoch}, best validation accuracy is {best_IOU}!"
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
        elif val_IOU <= best_IOU and epoch != args.epochs - 1:
            epochs_without_improvement += 1
            print(f"Epochs without Improvement: {epochs_without_improvement}")

            if epochs_without_improvement == 30:
                print(
                    f"{epochs_without_improvement} epochs without improvement, Stopping Training!"
                )
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VLN Navigation")
    parser.add_argument("--seed", default=420, type=int, help="random seed")
    parser.add_argument("--epochs", default=200, type=int, help="epoch size")

    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    parser.add_argument("--num_workers", type=int, default=10, help="number of workers")

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
        "--img_backbone",
        default="vit_tiny_patch16_224",
        choices=[
            "vit_tiny_patch16_224",
            "vit_small_patch16_224",
            "vit_tiny_patch16_384",
            "vit_small_patch16_384",
            "dino_resnet50"
        ],
        type=str,
    )
    parser.add_argument("--image_dim", type=int, default=224, help="Image Dimension")
    parser.add_argument("--mask_dim", type=int, default=112, help="Mask Dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden Dimension")

    parser.add_argument("--grad_check", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default="./saved_model")

    parser.add_argument("--threshold", type=float, default=0.4, help="mask threshold")

    parser.add_argument("--save", default=False, action="store_true")

    args = parser.parse_args()

    main(args)
