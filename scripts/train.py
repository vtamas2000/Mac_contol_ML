import argparse
import numpy
import torch
import torchvision
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import random_split, DataLoader
import yaml

import sys

sys.path.append("models")

from model import MotionDetect

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def get_args() -> argparse.Namespace:
    """
    Collects command line arguments for trainig the model
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        type=str,
        help="Path to the root data folder",
    )
    parser.add_argument(
        "-s", "--settings", required=True, help="Path to settings yaml file"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="Motion training",
        help="Name of MLFlow experiment",
    )
    parser.add_argument(
        "--run_name", type=str, default="Standard", help="Name of MLFlow run"
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()

    with open(args.settings) as file:
        settings = yaml.load(file, Loader=yaml.SafeLoader)

    dataset = torchvision.datasets.ImageFolder(
        args.directory, transform=torchvision.transforms.ToTensor()
    )
    train_dataset, val_dataset = random_split(
        dataset, [settings["train_size"], 1 - settings["train_size"]]
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=settings["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=settings["batch_size"], shuffle=True
    )

    model = MotionDetect(**settings["hyperparameters"])

    logger = MLFlowLogger(experiment_name=args.exp_name, run_name=args.run_name)

    logger.log_hyperparams(settings["hyperparameters"])

    trainer = L.Trainer(max_epochs=100, logger=logger, default_root_dir="saved_models")
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    trainer.save_checkpoint("saved_models/model.ckpt")


if __name__ == "__main__":
    main()
