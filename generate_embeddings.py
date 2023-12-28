import os
import csv
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset.e_piano import create_embedding_datasets, compute_epiano_accuracy

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

    ##### Datasets #####
    train_dataset, val_dataset, test_dataset = create_embedding_datasets(args.input_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    if(args.continue_weights is None):
        raise ValueError("-continue_weights must be specified")
    model.load_state_dict(torch.load(args.continue_weights))

    ##### Not smoothing evaluation loss #####
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    generate_embeddings(model, dataloaders, args.output_dir)


if __name__ == "__main__":
    main()
