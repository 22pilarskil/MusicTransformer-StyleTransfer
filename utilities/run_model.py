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

triplet_loss_fn = torch.nn.TripletMarginLoss(margin=0.2, p=2)
torch.autograd.set_detect_anomaly(True)

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
            total_loss = 0
            print(batch[3], feature_size)
            for num, sample in enumerate(batch[:3]):
                x = sample[0].to(get_device())
                tgt = sample[1].to(get_device())
                print(x.shape)      	        
                if True: #with autocast():
                    y, style_embedding, positional_embedding = model(x)
                    print(y.shape)
                    if feature_size:
                        style_embeddings.append(y)
                    else:
                        style_embeddings.append(style_embedding)
                        y = y.reshape(y.shape[0] * y.shape[1], -1)
                        tgt = tgt.flatten()
                        out = loss.forward(y, tgt)
                        total_loss += out  # Store the item for logging
                        del out
            triplet_loss = triplet_loss_fn(style_embeddings[0], style_embeddings[1], style_embeddings[2])
            if feature_size:
                combined_loss = triplet_loss
            else:
                combined_loss = total_loss + triplet_loss
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
        except ValueError as e:
            print(f"Skipping batch {batch_num+1} due to error: {e}")
            batch_num += 1
            skipped += 1
            continue  # Skip to the next batch

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
        sum_loss   = 0.0
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
                        y, style_embedding, positional_embedding = model(x)
                        if feature_size:
                            style_embeddings.append(y)
                        else:
                            style_embeddings.append(style_embedding)
                            sum_acc += float(compute_epiano_accuracy(y, tgt)) / num_samples
 
                            y = y.reshape(y.shape[0] * y.shape[1], -1)
                            tgt = tgt.flatten()
        
                            out = loss.forward(y, tgt)
                            sum_loss += float(out) / num_samples

                triplet_loss = triplet_loss_fn(style_embeddings[0], style_embeddings[1], style_embeddings[2])
                sum_triplet_loss += float(triplet_loss)

                batch_num += 1

            except StopIteration:
                break  # End of the dataset
            except ValueError as e:
                print(f"Skipping batch {batch_num+1} due to error: {e}")
                continue  # Skip to the next batch

        avg_loss    = sum_loss / n_test
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
        jazz_embeddings = []
        classical_embeddings = []
        while True:
            try:
                batch = next(dataloader_iter)
                num_samples = len(batch)
                style_embeddings = []
                style_embeddings_tensor = []
                for num, sample in enumerate(batch[:3]):

                    x = sample[0].to(get_device())
                    tgt = sample[1].to(get_device())

                    y, style_embedding, positional_embedding = model(x)
                    style_embeddings.append(y.flatten().cpu().numpy())
                    style_embeddings_tensor.append(y)
                    # style_embeddings.append(style_embedding.flatten().cpu().numpy())
                triplet_loss = triplet_loss_fn(style_embeddings_tensor[0], style_embeddings_tensor[1], style_embeddings_tensor[2])                
                print("TRIPLET LOSS", triplet_loss)

                if "jazz" in batch[3]:
                    jazz_embeddings.extend(style_embeddings[:2])
                    classical_embeddings.append(style_embeddings[2])
                elif "classical" in batch[3]:
                    classical_embeddings.extend(style_embeddings[:2])
                    jazz_embeddings.append(style_embeddings[2])
                else:
                    raise ValueError("Invalid label:", batch[3], batch_num)

                batch_num += 1
                print("BATCH_NUM", batch_num)
                if batch_num == iterations:
                    break

            except StopIteration:
                break  # End of the dataset
        jazz_embeddings_tensors = [torch.tensor(emb) for emb in jazz_embeddings]
        embedding_stack = torch.stack(jazz_embeddings_tensors)
        norms = torch.norm(embedding_stack, p=2, dim=1)
        variance = torch.var(norms)
        print("NORM", norms)
        print("VARIANCE", variance)

        classical_embeddings_tensors = [torch.tensor(emb) for emb in classical_embeddings]
        embedding_stack = torch.stack(classical_embeddings_tensors)
        norms = torch.norm(embedding_stack, p=2, dim=1)
        variance = torch.var(norms)
        print("NORM", norms)
        print("VARIANCE", variance)

        all_embeddings = np.concatenate([jazz_embeddings, classical_embeddings])
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, method="barnes_hut")
        reduced_embeddings = tsne.fit_transform(all_embeddings)
        #reduced_embeddings = all_embeddings
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(reduced_embeddings)
        jazz_reduced = reduced_embeddings[:len(jazz_embeddings)]
        classical_reduced = reduced_embeddings[len(jazz_embeddings):]

        plt.scatter(jazz_reduced[:, 0], jazz_reduced[:, 1], c='blue', label='Jazz')
        plt.scatter(classical_reduced[:, 0], classical_reduced[:, 1], c='red', label='Classical')

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

                        y, style_embedding, positional_embedding = model(x)

                        count += 1
                        file_path = os.path.join(output_dir, dataloader, f"{dataloader}_{count}.pkl")

                        with open(file_path, 'wb') as file:
                            pickle.dump([x, style_embedding, positional_embedding, negative_style, label], file)

                        print(f"File saved to {file_path} with label {label}")
                       

                except StopIteration:
                    break  # End of the dataset

    return


