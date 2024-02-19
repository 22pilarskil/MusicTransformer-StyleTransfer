import torch
import time
import gc
import os
import shutil
import pickle

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr

from dataset.e_piano import compute_epiano_accuracy

import torch.nn.functional as F
from torch.nn.functional import softmax, cosine_similarity
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)
torch.autograd.set_detect_anomaly(True)
similarity_coeff = 3.0
std_coeff = 50.0
limit = 10000
loss_scale = 1
add_loss_scale = 0

def inverse(x, n=0.75):
    return 1 / (1-x)**n

def compute_similarity(logits1, logits2, temperature=0.01, power=2):
    probs1 = softmax(logits1 / temperature, dim=-1) ** power
    probs2 = softmax(logits2 / temperature, dim=-1) ** power
    probs1 = probs1 / probs1.sum(dim=-1, keepdim=True)
    probs2 = probs2 / probs2.sum(dim=-1, keepdim=True)
    similarities = cosine_similarity(probs1, probs2, dim=-1)
    mean_similarity = similarities.mean()
    return mean_similarity

# eval_model
def eval_reconstructor_model(model, dataloader, loss_func):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """

    model.eval()

    avg_acc     = -1
    avg_loss    = -1
    avg_add_loss = -1
    batch_num = 0
    with torch.set_grad_enabled(False):
        sum_loss   = 0.0
        sum_acc    = 0.0
        sum_add_loss = 0.0
        dataloader_iter = iter(dataloader)
        while True:
            try:

                time_before = time.time()
                batch = next(dataloader_iter)

                x = batch[0]
                style_embedding = batch[1]
                content_embedding = batch[2]
                tgt = batch[3] 
                y = model(x, style_embedding, content_embedding)
                y   = y.reshape(y.shape[0] * y.shape[1], -1)
                tgt = tgt.flatten()

                out = loss_func.forward(y, tgt)

                new_style_embedding = torch.roll(style_embedding, shifts=1, dims=0)
                new_content_embedding = torch.roll(content_embedding, shifts=1, dims=0)
                with torch.no_grad():
                    # Forward pass with rotated style_embedding
                    new_y = model(x, new_style_embedding, new_content_embedding)  # Detach to avoid gradients for new_style_embedding
                    new_y = new_y.reshape(new_y.shape[0] * new_y.shape[1], -1)
                    y_normalized = F.normalize(y, p=2, dim=1)
                    new_y_normalized = F.normalize(new_y, p=2, dim=1)

                    cosine_sim = torch.sum(y_normalized * new_y_normalized, dim=1)
                    additional_loss = cosine_sim.mean()


                sum_acc += float(compute_epiano_accuracy(y, tgt))
                sum_add_loss += float(additional_loss)
                sum_loss += float(out)

                time_after = time.time()
                time_took = time_after - time_before
                
                batch_num += 1
#                print("BATCH:", batch_num, time_took)

            except StopIteration:
                break  # End of the dataset

            if batch_num == limit: break

        avg_loss    = sum_loss / batch_num
        avg_accuracy = sum_acc / batch_num
        avg_add_loss = sum_add_loss / batch_num

    return avg_loss, avg_accuracy, avg_add_loss




def train_reconstructor_epoch(cur_epoch, model, dataloader, loss_func, opt, lr_scheduler=None):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Trains a single model epoch
    ----------
    """

    out = -1
    model.train()
    batch_num = 0
    dataloader_iter = iter(dataloader)
    print("HERE")
    while True:
        try:

            batch = next(dataloader_iter)
            batch_num += 1
            time_before = time.time()

            opt.zero_grad()
            gc.collect()

            x = batch[0]
            style_embedding = batch[1]
            content_embedding = batch[2]
            tgt = batch[3]

            # Original forward pass
            y = model(x, style_embedding, content_embedding)
            y = y.reshape(y.shape[0] * y.shape[1], -1)
            tgt = tgt.flatten()
            print(tgt[:10])
            additional_loss = 0

            # Original loss
            loss = loss_func(y, tgt)
            print("GENERATION_LOSS", loss.item())
            # Rotate style_embedding
            new_style_embedding = torch.roll(style_embedding, shifts=1, dims=0)
            new_content_embedding = torch.roll(content_embedding, shifts=1, dims=0)
            with torch.no_grad():
                # Forward pass with rotated style_embedding
                new_y = model(x, new_style_embedding, new_content_embedding)  # Detach to avoid gradients for new_style_embedding
                new_y = new_y.reshape(new_y.shape[0] * new_y.shape[1], -1)
    
                print(torch.argmax(new_y,dim=-1)[:10])
                print(torch.argmax(y, dim=-1)[:10])

                y_normalized = F.normalize(y, p=2, dim=1)
                new_y_normalized = F.normalize(new_y, p=2, dim=1)

                cosine_sim = torch.sum(y_normalized * new_y_normalized, dim=1)
                additional_loss = inverse(cosine_sim.mean())

                print("ADDITIONAL_LOSS", additional_loss.item(), cosine_sim.mean())    
  
            # Combine the losses
            combined_loss = loss_scale * loss + add_loss_scale * additional_loss
            scaler.scale(combined_loss).backward()
            scaler.step(opt)
            scaler.update()

            # Gradient analysis

            if(lr_scheduler is not None):
                lr_scheduler.step()

            time_after = time.time()
            time_took = time_after - time_before

            print(SEPERATOR)
            print("TOTAL", float(combined_loss))
            print("Epoch", cur_epoch, " Batch", batch_num, "/", len(dataloader))
            print("LR:", get_lr(opt))
            print("Train loss:", float(loss), float(additional_loss))
            print("")
            print("Time (s):", time_took)
            print(SEPERATOR)
            print("")

            del loss #, combined_loss, additional_loss, y, new_y, new_y_probs

            torch.cuda.empty_cache()

        except StopIteration:
            print(f"EPOCH {cur_epoch} finished!")
            break  # End of the dataset
        
        if batch_num == limit: break        
        

    return
