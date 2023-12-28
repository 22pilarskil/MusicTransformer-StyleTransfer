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

from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
scaler = GradScaler()

MARGIN = 1.0
triplet_loss_fn = torch.nn.TripletMarginLoss(margin=MARGIN, p=2)
torch.autograd.set_detect_anomaly(True)



def create_triplet_mask(labels):
    labels = labels.unsqueeze(1)  # Shape: (batch_size, 1)
    mask_anchor_positive = labels == labels.T  # Anchor and Positive of
    mask_anchor_negative = labels != labels.T  # Anchor and Negative of

    valid_triplets = mask_anchor_positive.unsqueeze(2) & mask_anchor_negative.unsqueeze(1)
    valid_triplet_indices = valid_triplets.nonzero(as_tuple=False)
    valid_triplet_indices = valid_triplet_indices[valid_triplet_indices[:, 0] != valid_triplet_indices[:, 1]]
    return valid_triplet_indices


def compute_triplet_distances(embeddings, labels, margin, return_all=False):
    triplet_indices = create_triplet_mask(labels)
    anchor_embeddings = embeddings[triplet_indices[:, 0]]
    positive_embeddings = embeddings[triplet_indices[:, 1]]
    negative_embeddings = embeddings[triplet_indices[:, 2]]

    pos_dist = torch.nn.functional.pairwise_distance(anchor_embeddings, positive_embeddings, p=2)
    neg_dist = torch.nn.functional.pairwise_distance(anchor_embeddings, negative_embeddings, p=2)

    semi_hard_triplet_mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + margin)
    hard_triplet_mask = (pos_dist - neg_dist + margin > 0)

    zero_loss_triplets = ~(semi_hard_triplet_mask | (pos_dist - neg_dist + margin > 0))
    zero_loss_count = zero_loss_triplets.sum().item()
    print("Zero Loss Triplets:", zero_loss_count)

    if return_all:
        combined_mask = semi_hard_triplet_mask | hard_triplet_mask
        valid_indices = triplet_indices[combined_mask]
    else:
        valid_indices = triplet_indices[semi_hard_triplet_mask] if semi_hard_triplet_mask.any() else triplet_indices[hard_triplet_mask]

    print(f"Computing {len(valid_indices)} / {len(triplet_indices)}")
    return valid_indices

# train_epoch
def train_epoch(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1, feature_size=0):
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
    while True:
        try:
            batch = next(dataloader_iter)
            time_before = time.time()

            opt.zero_grad()
            gc.collect()
            style_embeddings = []
            labels = []
            total_loss = 0
            for num, sample in enumerate(batch):
                label = sample[2].to(get_device())
                x = sample[0].to(get_device())
                tgt = sample[1].to(get_device())

                with autocast():
                    y = model(x)
                    raise ValueError("StOP")
                    labels.append(label)
                    style_embeddings.append(y.to('cpu'))
            style_embeddings = [j.reshape(1, feature_size) for i in style_embeddings for j in i]
            labels = [j for i in labels for j in i]
            embeddings = torch.cat(style_embeddings, dim=0)
            labels = torch.tensor(labels)
            top_hard_triplets = compute_triplet_distances(embeddings, labels, MARGIN)

            top_anchor_embeddings = embeddings[top_hard_triplets[:, 0]]
            top_positive_embeddings = embeddings[top_hard_triplets[:, 1]]
            top_negative_embeddings = embeddings[top_hard_triplets[:, 2]]

            triplet_loss = triplet_loss_fn(top_anchor_embeddings, top_positive_embeddings, top_negative_embeddings)
            print(triplet_loss)
            combined_loss = triplet_loss.to("cuda:0")

            scaler.scale(combined_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            if(lr_scheduler is not None):
                lr_scheduler.step()

            time_after = time.time()
            time_took = time_after - time_before

            if((batch_num+1) % print_modulus == 0):
                print(SEPERATOR)
                print("Epoch", cur_epoch, " Batch", batch_num+1, "/", len(dataloader))
                print("LR:", get_lr(opt))
                print("Train loss:", float(total_loss))
                print("")
                print("Triplet loss:", float(triplet_loss))
                print("SKIPPED: ", skipped)
                print("Time (s):", time_took)
                print(SEPERATOR)
                print("")

            del total_loss
            del triplet_loss
            batch_num += 1

            torch.cuda.empty_cache()

        except StopIteration:
            print(f"EPOCH {cur_epoch} finished!")
            break  # End of the dataset

    return

# eval_model
def eval_model(model, dataloader, loss, feature_size=0):
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
        n_test      = len(dataloader)
        sum_loss   = []
        sum_acc    = 0.0
        sum_triplet_loss = 0.0
        dataloader_iter = iter(dataloader)
        while True:
            try:
                batch = next(dataloader_iter)
                num_samples = len(batch)
                style_embeddings = []
                for num, sample in enumerate(batch[:3]):
                    x = sample[0].to(get_device())
                    tgt = sample[1].to(get_device())
             
                    if True: #with autocast():
                        y = model(x)
                        style_embeddings.append(y)

                triplet_loss = triplet_loss_fn(style_embeddings[0], style_embeddings[1], style_embeddings[2])
                sum_triplet_loss += float(triplet_loss)

                batch_num += 1

            except StopIteration:
                break  # End of the dataset
        sum_loss = np.array(sum_loss)
        nan_count = np.sum(np.isnan(sum_loss))
        avg_loss = np.nansum(sum_loss) / ((n_test * 3) - nan_count)
        avg_acc     = sum_acc / n_test
        avg_triplet_loss = sum_triplet_loss / n_test

    return avg_loss, avg_acc, avg_triplet_loss


def eval_triplets(model, dataloader, iterations=40):
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
        n_test      = len(dataloader)
        sum_loss   = 0.0
        sum_acc    = 0.0
        sum_triplet_loss = 0.0
        dataloader_iter = iter(dataloader)
        embeddings = {}
        while True:
            try:
                batch = next(dataloader_iter)
                num_samples = len(batch)
                style_embeddings = []
                style_embeddings_tensor = []
                for num, sample in enumerate(batch[:3]):

                    x = sample[0].to(get_device())
                    tgt = sample[1].to(get_device())
                    print(x.shape)
                    y = model(x)
                    style_embeddings.append(y.flatten().cpu().numpy())
                    style_embeddings_tensor.append(y)
                    # style_embeddings.append(style_embedding.flatten().cpu().numpy())
                triplet_loss = triplet_loss_fn(style_embeddings_tensor[0], style_embeddings_tensor[1], style_embeddings_tensor[2])                
                print("TRIPLET LOSS", triplet_loss)

                positive = batch[3][0]
                negative = batch[4][0]
                if not positive in embeddings:
                    embeddings[positive] = []
                if not negative in embeddings:
                    embeddings[negative] = []
                
                embeddings[positive].extend(style_embeddings[:2])
                embeddings[negative].extend([style_embeddings[2]])

                batch_num += 1
                print("BATCH_NUM", batch_num)
                if batch_num == iterations:
                    break

            except StopIteration:
                break  # End of the dataset

        for key, value in embeddings.items():
            print(f"{key}: {np.array(value).shape}")
        all_embeddings = np.concatenate(list(embeddings.values()))
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, method="barnes_hut")
        reduced_embeddings = tsne.fit_transform(all_embeddings)

        kmeans = KMeans(n_clusters=len(embeddings))  # Number of clusters equals the number of classes
        kmeans.fit(reduced_embeddings)

# Color palette
        colors = plt.cm.rainbow(np.linspace(0, 1, len(embeddings)))

        start_idx = 0
        for (label, embedding_list), color in zip(embeddings.items(), colors):
    # Determine the length of the current class embeddings
            class_len = len(embedding_list)
            class_reduced = reduced_embeddings[start_idx:start_idx + class_len]
            start_idx += class_len
 
            plt.scatter(class_reduced[:, 0], class_reduced[:, 1], c=[color], label=label)

        if 'kmeans' in locals():
            centers = kmeans.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
 
        silhouette_avg = silhouette_score(reduced_embeddings, kmeans.labels_)
        # Normalizing the score to be between 0 and 100
        plt.legend()
        plt.title("Style Embedding Clusters - Score {}".format(silhouette_avg))
        plt.savefig('style_embedding_clusters.png')



    return




def generate_embeddings(model, dataloaders, output_dir):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """

    model.eval()

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    for dataloader in dataloaders:
        os.mkdir(os.path.join(output_dir, dataloader))
        count = 0
        with torch.set_grad_enabled(False):
            dataloader_iter = iter(dataloaders[dataloader])
            while True:
                try:
                    batch = next(dataloader_iter)
                    num_samples = len(batch)
                    _, negative_style, _ = model(batch[2][0].to(get_device()))
                    label = "classical" if "classical_pos" in batch[3] else "jazz"
                    for num, sample in enumerate(batch[:2]):

                        x = sample[0].to(get_device())
                        tgt = sample[1].to(get_device())

                        y = model(x)

                        count += 1
                        file_path = os.path.join(output_dir, dataloader, f"{dataloader}_{count}.pkl")
                        raise ValueError("STOP")
                        with open(file_path, 'wb') as file:
                            pickle.dump([x, style_embedding, positional_embedding, negative_style, label], file)

                        print(f"File saved to {file_path} with label {label}")
                       

                except StopIteration:
                    break  # End of the dataset

    return


