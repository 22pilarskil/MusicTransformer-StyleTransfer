import os
import csv
import shutil
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.multiprocessing as mp

from dataset.e_piano import create_embedding_datasets, create_epiano_datasets, compute_epiano_accuracy

from model.music_transformer import MusicTransformer
from model.reconstructor import Reconstructor
from model.loss import SmoothCrossEntropyLoss
from embedding_loss import EmbeddingLoss 

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_train_reconstruction_args, print_train_reconstruction_args, write_model_params
from utilities.run_reconstructor import train_reconstructor_epoch, eval_reconstructor_model

from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

CSV_HEADER = ["Epoch", "Learn rate", "Avg Train loss", "Avg Train Add loss", "Avg Train acc", "Avg Eval loss", "Avg Eval Add loss", "Avg Eval acc"]

# Baseline is an untrained epoch that we evaluate as a baseline loss and accuracy
BASELINE_EPOCH = -1
mp.set_start_method('spawn', force=True)
start = 1

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Trains a model specified by command line arguments
    ----------
    """

    args = parse_train_reconstruction_args()
    print_train_reconstruction_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    ##### Output prep #####
    params_file = os.path.join(args.output_dir, "model_params.txt")
    write_model_params(args, params_file)

    weights_folder = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_folder, exist_ok=True)

    results_folder = os.path.join(args.output_dir, "results")
    os.makedirs(results_folder, exist_ok=True)

    results_file = os.path.join(results_folder, "results.csv")
    best_loss_file = os.path.join(results_folder, "best_loss_weights.pickle")
    best_acc_file = os.path.join(results_folder, "best_acc_weights.pickle")
    best_text = os.path.join(results_folder, "best_epochs.txt")

    ##### Datasets #####
    train_dataset, val_dataset, test_dataset = create_embedding_datasets(args.input_dir, args.max_sequence)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    train_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    eval_loss_func = train_loss_func

    model = Reconstructor(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    ##### Continuing from previous training session #####
    start_epoch = BASELINE_EPOCH + start
    if (args.continue_weights is not None):
        if (args.continue_epoch is None):
            print("ERROR: Need epoch number to continue from (-continue_epoch) when using continue_weights")
            return
        else:
            model.load_state_dict(torch.load(args.continue_weights))
            start_epoch = args.continue_epoch
    elif (args.continue_epoch is not None):
        print("ERROR: Need continue weights (-continue_weights) when using continue_epoch")
        return

    if(args.lr is None):
        if(args.continue_epoch is None):
            init_step = 0
        else:
            init_step = args.continue_epoch * len(train_loader)

        lr = LR_DEFAULT_START
        lr_stepper = LrStepTracker(args.d_model, SCHEDULER_WARMUP_STEPS, init_step)
    else:
        lr = args.lr


    #print(model.linear_embeddings.weight)
    #print(model.linear_embeddings.weight.shape)
    #raise ValueError()
    ##### Optimizer #####
    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

    if(args.lr is None):
        lr_scheduler = LambdaLR(opt, lr_stepper.step)
    else:
        lr_scheduler = None

    ##### Tracking best evaluation accuracy #####
    best_eval_acc        = 0.0
    best_eval_acc_epoch  = -1
    best_eval_loss       = float("inf")
    best_eval_loss_epoch = -1

    ##### Results reporting #####
    if(not os.path.isfile(results_file)):
        with open(results_file, "w", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(CSV_HEADER)


    ##### TRAIN LOOP #####
    for epoch in range(start_epoch, args.epochs):
        # Baseline has no training and acts as a base loss and accuracy (epoch 0 in a sense)
        if(epoch > BASELINE_EPOCH):
            print(SEPERATOR)
            print("NEW EPOCH:", epoch+1)
            print(SEPERATOR)
            print("")

            # Train
            train_reconstructor_epoch(epoch+1, model, train_loader, train_loss_func, opt, lr_scheduler)

            print(SEPERATOR)
            print("Evaluating:")
        else:
            print(SEPERATOR)
            print("Baseline model evaluation (Epoch 0):")

        # Eval
        train_loss, train_acc, train_add_loss = eval_reconstructor_model(model, train_loader, train_loss_func)
        eval_loss, eval_acc, eval_add_loss = eval_reconstructor_model(model, test_loader, eval_loss_func)

        # Learn rate
        lr = get_lr(opt)

        print("Epoch:", epoch+1)
        print("Avg train loss:", train_loss)
        print("Avg eval loss:", eval_loss)
        print("Avg train add loss:", train_add_loss)
        print("Avg eval add loss:", eval_add_loss)
        print("Avg train acc:", train_acc)
        print("Avg eval acc:", eval_acc)
        print(SEPERATOR)
        print("")

        new_best = False

        if(eval_loss < best_eval_loss):
            best_eval_loss       = eval_loss
            best_eval_loss_epoch = epoch+1
            torch.save(model.state_dict(), best_loss_file)
            new_best = True

        if(eval_acc < best_eval_acc):
            best_eval_acc       = eval_acc
            best_eval_acc_epoch = epoch+1
            torch.save(model.state_dict(), best_acc_file)
            new_best = True

        # Writing out new bests
        if(new_best):
            with open(best_text, "w") as o_stream:
                print("Best eval loss epoch:", best_eval_loss_epoch, file=o_stream)
                print("Best eval loss:", best_eval_loss, file=o_stream)
                print("Best eval acc epoch:", best_eval_acc_epoch, file=o_stream)
                print("Best eval acc:", best_eval_acc, file=o_stream)



        if((epoch+1) % args.weight_modulus == 0):
            epoch_str = str(epoch+1).zfill(PREPEND_ZEROS_WIDTH)
            path = os.path.join(weights_folder, "epoch_" + epoch_str + ".pickle")
            torch.save(model.state_dict(), path)

        with open(results_file, "a", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow([epoch+1, lr, train_loss, train_add_loss, train_acc, eval_loss, eval_add_loss, eval_acc])


    return


if __name__ == "__main__":
    main()
