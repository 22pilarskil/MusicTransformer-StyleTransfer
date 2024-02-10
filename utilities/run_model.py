import time
import gc
import os
import shutil
import pickle

from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr

import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from .tools import compute_triplet_distances, compute_average_pairwise_distances

scaler = GradScaler()

TRIPLET_MARGIN = 1.0
triplet_loss_fn = torch.nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
CONTRASTIVE_MARGIN = 1.0
torch.autograd.set_detect_anomaly(True)
limit = 1000
return_all = True

# train_epoch

def train_epoch_style(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1, feature_size=0):

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

                y = model(x)
                labels.append(label)
                style_embeddings.append(y.to('cpu'))

            style_embeddings = [j.reshape(1, feature_size) for i in style_embeddings for j in i]
            labels = [j.to("cpu").unsqueeze(dim=-1) for i in labels for j in i]
            embeddings = torch.cat(style_embeddings, dim=0)
            print("LABELS", len(labels), len(style_embeddings))

            top_hard_triplets = compute_triplet_distances(embeddings, labels, TRIPLET_MARGIN, return_all=return_all)

            top_anchor_embeddings = embeddings[top_hard_triplets[:, 0]]
            top_positive_embeddings = embeddings[top_hard_triplets[:, 1]]
            top_negative_embeddings = embeddings[top_hard_triplets[:, 2]]

            triplet_loss = triplet_loss_fn(top_anchor_embeddings, top_positive_embeddings, top_negative_embeddings)
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


def train_epoch_content(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1, feature_size=0):

    model.train()
    batch_num = 0
    skipped = 0
    dataloader_iter = iter(dataloader)
    while True:
        try:
            content_embeddings = []
            labels = []
            batch = next(dataloader_iter)
            time_before = time.time()

            opt.zero_grad()
            gc.collect()
            total_loss = 0

            y_melody = model(batch[0].to(get_device())).to("cpu")
            y_harmony = model(batch[1].to(get_device())).to("cpu")
            y_combined = model(batch[2].to(get_device())).to("cpu")

            split_melody = [y_melody[i].unsqueeze(0) for i in range(y_melody.size(0))]
            split_harmony = [y_harmony[i].unsqueeze(0) for i in range(y_harmony.size(0))]
            split_combined = [y_combined[i].unsqueeze(0) for i in range(y_combined.size(0))]

            content_embeddings = split_melody + split_harmony + split_combined
            embeddings = torch.cat(content_embeddings, dim=0)

            batch_size = y_melody.size(0)
            labels = []

            labels.extend([torch.Tensor([2 * i]).to("cpu").int() for i in range(batch_size)])
            labels.extend([torch.Tensor([2 * i + 1]).to("cpu").int() for i in range(batch_size)])
            labels.extend([torch.Tensor([2 * i, 2 * i + 1]).to("cpu").int() for i in range(batch_size)])

            print("LABELS", len(labels), len(content_embeddings))

            top_hard_triplets = compute_triplet_distances(embeddings, labels, TRIPLET_MARGIN, return_all=return_all)

            top_anchor_embeddings = embeddings[top_hard_triplets[:, 0]]
            top_positive_embeddings = embeddings[top_hard_triplets[:, 1]]
            top_negative_embeddings = embeddings[top_hard_triplets[:, 2]]

            triplet_loss = triplet_loss_fn(top_anchor_embeddings, top_positive_embeddings, top_negative_embeddings)
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
                print("Combined loss:", float(combined_loss))
                print("SKIPPED: ", skipped)
                print("Time (s):", time_took)
                print(SEPERATOR)
                print("")

            del combined_loss
            batch_num += 1

            torch.cuda.empty_cache()
           
            if batch_num > limit: break

        except StopIteration:
            print(f"EPOCH {cur_epoch} finished!")
            break  # End of the dataset

    return


def eval_model_style(model, dataloader, loss, feature_size=0):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """

    model.eval()
    batch_num = 0
    with torch.set_grad_enabled(False):
        n_test = len(dataloader)
        sum_loss = []
        sum_acc = 0.0
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
                    with autocast():
                        y = model(x)
                        style_embeddings.append(y)

                triplet_loss = triplet_loss_fn(style_embeddings[0], style_embeddings[1], style_embeddings[2])
                sum_triplet_loss += float(triplet_loss)
              
                batch_num += 1
                print("BATCH", batch_num)
                if batch_num > limit: break


            except StopIteration:
                break  # End of the dataset
        sum_loss = np.array(sum_loss)
        nan_count = np.sum(np.isnan(sum_loss))
        avg_loss = np.nansum(sum_loss) / ((n_test * 3) - nan_count)
        avg_acc = sum_acc / n_test
        avg_triplet_loss = sum_triplet_loss / n_test

    return avg_loss, avg_acc, avg_triplet_loss


def eval_model_content(model, dataloader, loss, feature_size=0):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """

    model.eval()
    batch_num = 0
    with torch.set_grad_enabled(False):
        n_test      = len(dataloader)
        sum_loss   = []
        sum_acc    = 0.0
        sum_combined_loss = 0.0
        sum_melody_harmony = 0.0
        sum_melody_combined = 0.0
        sum_harmony_combined = 0.0
        dataloader_iter = iter(dataloader)
        while True:
            try:
                batch = next(dataloader_iter)

                y_melody = model(batch[0].to(get_device())).to("cpu")
                y_harmony = model(batch[1].to(get_device())).to("cpu")
                y_combined = model(batch[2].to(get_device())).to("cpu")

                melody_harmony, melody_combined, harmony_combined = compute_average_pairwise_distances(y_melody, y_harmony, y_combined)
        
                sum_melody_harmony += melody_harmony
                sum_melody_combined += melody_combined
                sum_harmony_combined += harmony_combined
        
                batch_num += 1
                print("BATCH_NUM", batch_num, round(float(melody_harmony), 3), round(float(melody_combined), 3), round(float(harmony_combined), 3))
                if batch_num > limit:
                    print("BREAKING")
                    break

            except StopIteration:
                break  # End of the dataset

        avg_melody_harmony = sum_melody_harmony / batch_num
        avg_melody_combined = sum_melody_combined / batch_num
        avg_harmony_combined = sum_harmony_combined / batch_num
        return avg_melody_harmony, avg_melody_combined, avg_harmony_combined


def eval_triplets(model, dataloader, iterations=40, file_path="style_embeddings_clusters.png"):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """

    model.eval()
    plt.clf()
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
                    print(y)
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

        colors = plt.cm.rainbow(np.linspace(0, 1, len(embeddings)))

        start_idx = 0
        for (label, embedding_list), color in zip(embeddings.items(), colors):
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
        plt.savefig(file_path)
        plt.close()
        return 0, 0, silhouette_avg




def generate_embeddings(style_model, content_model, dataloaders, output_dir):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """

    style_model.eval()
    content_model.eval()

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

                    x = batch[0].to(get_device())
                    tgt = batch[1].to(get_device())

                    style_embedding = style_model(x)
                    content_embedding = content_model(x)
                    print(style_embedding.shape)
                    count += 1
                    file_path = os.path.join(output_dir, dataloader, f"{dataloader}_{count}.pkl")
                    with open(file_path, 'wb') as file:
                        pickle.dump([x, style_embedding, content_embedding, tgt], file)

                    print(f"File saved to {file_path}")

                except StopIteration:
                    break  # End of the dataset

    return


