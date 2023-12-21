import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utilities.constants import *
from utilities.device import cpu_device

SEQUENCE_START = 0

# EPianoDataset
class EPianoDataset(Dataset):
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

class oldEPianoDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.
    ----------
    """

    def __init__(self, root, max_seq=2048, random_seq=True):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]
        #self.data_files = [self.data_files[0]]

    # __len__
    def __len__(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        How many data files exist in the given directory
        ----------
        """

        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.

        Returns the input and the target.
        ----------
        """

        # All data on cpu to allow for the Dataloader to multithread
        i_stream    = open(self.data_files[idx], "rb")
        # return pickle.load(i_stream), None
        raw_data = pickle.load(i_stream)
        if len(raw_data) != 4:
            raw_mid     = torch.tensor(raw_data, dtype=TORCH_LABEL_TYPE, device=cpu_device())
            i_stream.close()
            x, tgt = process_midi(raw_mid, self.max_seq, self.random_seq)
            return [[x, tgt]]
        raw_mid     = torch.tensor(raw_data[:3], dtype=TORCH_LABEL_TYPE, device=cpu_device())
        i_stream.close()
        vals = []
        for mid in raw_mid[:3]:
            x, tgt = process_midi(mid, self.max_seq, self.random_seq)
            vals.append([x,tgt])
        vals.append(raw_data[3])

        return vals


class EmbeddingsDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.
    ----------
    """

    def __init__(self, root):
        self.root       = root

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]

    # __len__
    def __len__(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        How many data files exist in the given directory
        ----------
        """

        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.

        Returns the input and the target.
        ----------
        """

        # All data on cpu to allow for the Dataloader to multithread
        i_stream    = open(self.data_files[idx], "rb")
        # return pickle.load(i_stream), None
        raw_data = pickle.load(i_stream)
        i_stream.close()
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


# create_epiano_datasets
def create_epiano_datasets(dataset_root, max_seq, random_seq=True):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Creates train, evaluation, and test EPianoDataset objects for a pre-processed (preprocess_midi.py)
    root containing train, val, and test folders.
    ----------
    """

    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = EPianoDataset(train_root, max_seq, random_seq)
    val_dataset = EPianoDataset(val_root, max_seq, random_seq)
    test_dataset = EPianoDataset(test_root, max_seq, random_seq)

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