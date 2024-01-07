import os
import csv
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy

from model.music_transformer import MusicTransformer
from model.loss import SmoothCrossEntropyLoss

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params
from utilities.run_model import generate_embeddings



# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Trains a model specified by command line arguments
    ----------
    """

    args = parse_train_args()
    print_train_args(args)

    style_model_path = "../results_style/weights/epoch_0048.pickle"
    content_model_path = "../results_content/weights/epoch_0052.pickle"
    ##### Datasets #####
    train_dataset, val_dataset, test_dataset = create_epiano_datasets(args.input_dir, args.max_sequence)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    style_model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                feature_size=args.feature_size, max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    content_model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                feature_size=args.feature_size, max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())


    style_model.load_state_dict(torch.load(style_model_path))
    content_model.load_state_dict(torch.load(content_model_path))

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    generate_embeddings(style_model, content_model, dataloaders, args.output_dir)


if __name__ == "__main__":
    main()
