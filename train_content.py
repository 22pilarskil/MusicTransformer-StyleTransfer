import os
import csv
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset.e_piano import create_separated_datasets

from model.music_transformer import MusicTransformer
from model.loss import SmoothCrossEntropyLoss

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params
from utilities.run_model import train_epoch_content, eval_model_content

CSV_HEADER = ["Epoch", "Learn rate", "train_melody_harmony", "train_melody_combined", "train_harmony_combined", "eval_melody_harmony", "eval_melody_combined",
              "eval_harmony_combined"]

start = 0
# Baseline is an untrained epoch that we evaluate as a baseline loss and accuracy
BASELINE_EPOCH = -1


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

    if (args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

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

    ##### Tensorboard #####
    if (args.no_tensorboard):
        tensorboard_summary = None
    else:
        from torch.utils.tensorboard import SummaryWriter

        tensorboad_dir = os.path.join(args.output_dir, "tensorboard")
        tensorboard_summary = SummaryWriter(log_dir=tensorboad_dir)

    ##### Datasets #####
    train_dataset, test_dataset, val_dataset = create_separated_datasets(args.input_dir, args.max_sequence)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                             feature_size=args.feature_size, d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                             dropout=args.dropout,
                             transition_features=args.transition_features, reduction_factor=args.reduction_factor,
                             max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    def check_grads(model):
        def hook_fn(grad):
            if torch.isnan(grad).any():
                print("NaN value in gradient")

        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(hook_fn)

    check_grads(model)

    ##### Continuing from previous training session #####
    start_epoch = start + BASELINE_EPOCH
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

    ##### Lr Scheduler vs static lr #####
    if (args.lr is None):
        if (args.continue_epoch is None):
            init_step = 0
        else:
            init_step = args.continue_epoch * len(train_loader)

        lr = LR_DEFAULT_START
        lr_stepper = LrStepTracker(args.d_model, SCHEDULER_WARMUP_STEPS, init_step)
    else:
        lr = args.lr

    ##### Not smoothing evaluation loss #####
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    ##### SmoothCrossEntropyLoss or CrossEntropyLoss for training #####
    if (args.ce_smoothing is None):
        train_loss_func = eval_loss_func
    else:
        train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE, ignore_index=TOKEN_PAD)

    ##### Optimizer #####
    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

    if (args.lr is None):
        lr_scheduler = LambdaLR(opt, lr_stepper.step)
    else:
        lr_scheduler = None

    ##### Tracking best evaluation accuracy #####
    best_eval_acc = 0.0
    best_eval_acc_epoch = -1
    best_eval_loss = float("inf")
    best_eval_loss_epoch = -1

    ##### Results reporting #####
    if (not os.path.isfile(results_file)):
        with open(results_file, "w", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(CSV_HEADER)

    ##### TRAIN LOOP #####
    for epoch in range(start_epoch, args.epochs):
        # Baseline has no training and acts as a base loss and accuracy (epoch 0 in a sense)
        if (epoch > BASELINE_EPOCH):
            print(SEPERATOR)
            print("NEW EPOCH:", epoch + 1)
            print(SEPERATOR)
            print("")

            # Train
            train_epoch_content(epoch + 1, model, train_loader, train_loss_func, opt, lr_scheduler, args.print_modulus,
                              args.feature_size)

            print(SEPERATOR)
            print("Evaluating:")
        else:
            print(SEPERATOR)
            print("Baseline model evaluation (Epoch 0):")

        # Eval
        train_melody_harmony, train_melody_combined, train_harmony_combined = eval_model_content(model, train_loader, train_loss_func,
                                                                     args.feature_size)
        eval_melody_harmony, eval_melody_combined, eval_harmony_combined = eval_model_content(model, test_loader, eval_loss_func,
                                                                  args.feature_size)

        # Learn rate
        lr = get_lr(opt)

        print("Epoch:", epoch + 1)
        print("Avg train_melody_harmony:", train_melody_harmony)
        print("Avg train_melody_combined:", train_melody_combined)
        print("Avg train_harmony_combined:", train_harmony_combined)
        print("Avg eval_melody_harmony:", eval_melody_harmony)
        print("Avg eval_melody_combined:", eval_melody_combined)
        print("Avg eval_harmony_combined:", eval_harmony_combined)
        print(SEPERATOR)
        print("")


        if ((epoch + 1) % args.weight_modulus == 0):
            epoch_str = str(epoch + 1).zfill(PREPEND_ZEROS_WIDTH)
            path = os.path.join(weights_folder, "epoch_" + epoch_str + ".pickle")
            torch.save(model.state_dict(), path)

        with open(results_file, "a", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(
                [epoch + 1, lr, train_melody_harmony, train_melody_combined, train_harmony_combined, eval_melody_harmony, eval_melody_combined, eval_harmony_combined])

    # Sanity check just to make sure everything is gone
    if (not args.no_tensorboard):
        tensorboard_summary.flush()

    return


if __name__ == "__main__":
    main()
