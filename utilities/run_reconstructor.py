import torch
import time
import gc
import os
import shutil
import pickle

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

from torch.nn.functional import softmax, cosine_similarity
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2)
torch.autograd.set_detect_anomaly(True)
similarity_coeff = 3.0
std_coeff = 50.0

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
    batch_num = 0
    with torch.set_grad_enabled(False):
        sum_loss   = 0.0
        sum_acc    = 0.0
        sum_triplet_loss = 0.0
        dataloader_iter = iter(dataloader)
        while True:
            try:
                batch = next(dataloader_iter)

                style_emb = torch.squeeze(batch[1], 0)
                pos_emb = torch.squeeze(batch[2], 0)
                input_sequence = torch.squeeze(batch[0])
                if True: #with autocast():
                    logits_encoded = torch.squeeze(model(style_emb, pos_emb))
                    out = loss_func.forward(logits_encoded, input_sequence)
                    output_sequence = torch.argmax(logits_encoded, dim=1)
                    accuracy = (input_sequence == output_sequence).sum().item() / len(input_sequence)

                sum_acc += accuracy
                sum_loss += float(out)


                batch_num += 1
                print("BATCH:", batch_num)

            except StopIteration:
                break  # End of the dataset
            except ValueError as e:
                print(f"Skipping batch {batch_num+1} due to error: {e}")
                continue  # Skip to the next batch
            if batch_num == 1000: break

        avg_loss    = sum_loss / batch_num
        avg_accuracy = sum_acc / batch_num

    return avg_loss, avg_accuracy




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
    skipped = 0
    dataloader_iter = iter(dataloader)
    print("HERE")
    while True:
        try:
            batch = next(dataloader_iter)
            batch_num += 1
            time_before = time.time()

            opt.zero_grad()
            gc.collect()

            style_emb = torch.squeeze(batch[1], 0)
            pos_emb = torch.squeeze(batch[2], 0)
            neg_style_emb = torch.squeeze(batch[3], 0)
            input_sequence = torch.squeeze(batch[0])
            print(style_emb.shape)
            print(pos_emb.shape)
            if True: # with autocast():
                logits_encoded = torch.squeeze(model(style_emb, pos_emb))            
                print(logits_encoded.shape)
                out = loss_func.forward(logits_encoded, input_sequence)

            if True: # with autocast():
                negative_logits_encoded = torch.squeeze(model(neg_style_emb, pos_emb))

            similarity_penalty = compute_similarity(logits_encoded, negative_logits_encoded)

            total_loss = out + similarity_coeff * similarity_penalty #+ std_coeff / negative_logits_encoded.std()

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if(lr_scheduler is not None):
                lr_scheduler.step()

            time_after = time.time()
            time_took = time_after - time_before

            print(SEPERATOR)
            print("TOTAL", float(out))
            print("SIMILARITY", float(similarity_penalty), float(similarity_penalty * similarity_coeff))
            print("Epoch", cur_epoch, " Batch", batch_num, "/", len(dataloader))
            print("LR:", get_lr(opt))
            print("Train loss:", float(total_loss))
            print("")
            print("SKIPPED: ", skipped)
            print("Time (s):", time_took)
            print(SEPERATOR)
            print("")


            torch.cuda.empty_cache()

        except StopIteration:
            print(f"EPOCH {cur_epoch} finished!")
            break  # End of the dataset
        
        except ValueError as e:
            print(f"Skipping batch {batch_num+1} due to error: {e}")
            batch_num += 1
            skipped += 1
            continue  # Skip to the next batch
        if batch_num == 1000: break        
        

    return
