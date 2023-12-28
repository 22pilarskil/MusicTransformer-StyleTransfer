import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utilities.constants import *
from utilities.device import cpu_device

SEQUENCE_START = 0

class RandomizedBatching(Dataset):
    def __init__(self, root_path, max_seq=2048, random_seq=True):
        self.max_seq = max_seq
        self.class_directories = {d: os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))}
        self.class_files = {class_name: [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))] for class_name, dir_path in self.class_directories.items()}
        self.used_files = set()

    def __len__(self):
        return min(len(files) for files in self.class_files.values())
    
    def __getitem__(self, idx):

        class_choices = list(self.class_files.keys())
        positive_class = random.choice(class_choices)
        class_choices.remove(positive_class)
        negative_class = random.choice(class_choices)

        positive_file = self._select_file(self.class_files[positive_class])
        negative_file = self._select_file(self.class_files[negative_class])

        positive_id = list(self.class_files.keys()).index(positive_class)
        negative_id = list(self.class_files.keys()).index(negative_class)
        positive_sample = self._process_file(positive_file) + [positive_id]
        negative_sample = self._process_file(negative_file) + [negative_id]
        sample = [positive_sample, negative_sample] 
        return sample

    def _select_file(self, file_list):
        available_files = list(set(file_list) - self.used_files)
        if not available_files:
            # Reset if all files have been used
            self.used_files.clear()
            available_files = file_list

        selected_file = random.choice(available_files)
        self.used_files.add(selected_file)
        return selected_file

    def _process_file(self, file_path):
        i_stream    = open(file_path, "rb")
        raw_data = pickle.load(i_stream)
        i_stream.close()
        raw_data = torch.tensor(raw_data, dtype=TORCH_LABEL_TYPE, device=cpu_device())
        # Assuming process_midi is a function to process your midi data
        x, tgt = process_midi(raw_data, self.max_seq)
        crop = random.randint(0, 250)
        buffer = torch.Tensor([TOKEN_END for i in range(crop)]).int()
        x, tgt = torch.cat((x[crop:], buffer)), torch.cat((tgt[crop:], buffer))
        return [x, tgt]


# EPianoDataset
class TripletSelector(Dataset):
    def __init__(self, root_path, max_seq=2048, random_seq=True):
        self.max_seq = max_seq
        self.class_directories = {d: os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))}
        self.class_files = {class_name: [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))] for class_name, dir_path in self.class_directories.items()}
        self.used_files = set()

    def __len__(self):
        return min(len(files) for files in self.class_files.values())
    
    def __getitem__(self, idx):
        triplet = []

        class_choices = list(self.class_files.keys())
        positive_class = random.choice(class_choices)
        class_choices.remove(positive_class)
        negative_class = random.choice(class_choices)

        target_file = self._select_file(self.class_files[positive_class])
        positive_file = self._select_file(self.class_files[positive_class])
        negative_file = self._select_file(self.class_files[negative_class])

        triplet.append(self._process_file(target_file))
        triplet.append(self._process_file(positive_file))
        triplet.append(self._process_file(negative_file))
        triplet.append(positive_class)
        triplet.append(negative_class)
        return triplet

    def _select_file(self, file_list):
        available_files = list(set(file_list) - self.used_files)
        if not available_files:
            # Reset if all files have been used
            self.used_files.clear()
            available_files = file_list

        selected_file = random.choice(available_files)
        self.used_files.add(selected_file)
        return selected_file

    def _process_file(self, file_path):
        i_stream    = open(file_path, "rb")
        raw_data = pickle.load(i_stream)
        i_stream.close()
        raw_data = torch.tensor(raw_data, dtype=TORCH_LABEL_TYPE, device=cpu_device())
        # Assuming process_midi is a function to process your midi data
        x, tgt = process_midi(raw_data, self.max_seq)
        return [x, tgt]


class EmbeddingsDataset(Dataset):

    def __init__(self, root):
        self.root       = root
         
        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]
        print("LENGTH", len(self.data_files))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):

        i_stream    = open(self.data_files[idx], "rb")
        raw_data = pickle.load(i_stream)
        i_stream.close()
        print(len(raw_data))
        return raw_data



# process_midi
def process_midi(raw_mid, max_seq, random_seq=True):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    """

    x   = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    tgt = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq

    if(raw_len == 0):
        return x, tgt

    if(raw_len < full_seq):
        x[:raw_len]         = raw_mid
        tgt[:raw_len-1]     = raw_mid[1:]
        tgt[raw_len-1]      = TOKEN_END
    else:
        # Randomly selecting a range
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]


    # print("x:",x)
    # print("tgt:",tgt)

    return x, tgt


def create_triplet_datasets(dataset_root, max_seq, random_seq=True):

    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = TripletSelector(train_root, max_seq, random_seq)
    val_dataset = TripletSelector(val_root, max_seq, random_seq)
    test_dataset = TripletSelector(test_root, max_seq, random_seq)

    return train_dataset, val_dataset, test_dataset

def create_randomized_batching_datasets(dataset_root, max_seq, random_seq=True):

    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = RandomizedBatching(train_root, max_seq, random_seq)
    val_dataset = RandomizedBatching(val_root, max_seq, random_seq)
    test_dataset = RandomizedBatching(test_root, max_seq, random_seq)

    return train_dataset, val_dataset, test_dataset


def create_embedding_datasets(dataset_root):

    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = EmbeddingsDataset(train_root)
    val_dataset = EmbeddingsDataset(val_root)
    test_dataset = EmbeddingsDataset(test_root)

    return train_dataset, val_dataset, test_dataset


# compute_epiano_accuracy
def compute_epiano_accuracy(out, tgt):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Computes the average accuracy for the given input and output batches. Accuracy uses softmax
    of the output.
    ----------
    """

    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]

    # Empty
    if(len(tgt) == 0):
        return 1.0

    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    acc = num_right / len(tgt)

    return acc
