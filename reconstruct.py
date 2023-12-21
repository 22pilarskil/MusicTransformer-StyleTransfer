import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi, encode_midi

from utilities.argument_funcs import parse_reconstruct_args
from model.reconstructor import Reconstructor
from dataset.e_piano import create_embedding_datasets
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda

import encoding as our_encoding

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    args = parse_reconstruct_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    train_dataset, val_dataset, test_dataset = create_embedding_datasets(args.midi_root)
    dataset = val_dataset

    style_index = int(args.style_file)
    positional_index = int(args.positional_file)
    print("USING POSITIONAL INDEX {}, STYLE INDEX {}".format(positional_index, style_index))

    style_embedding = dataset[style_index][1]
    positional_embedding = dataset[positional_index][2]

    model = Reconstructor(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    model.load_state_dict(torch.load(args.model_weights, map_location=torch.device('cpu')))

    # f_path = os.path.join(args.output_dir, "original{}_{}.mid".format(positional_index, style_index))
    # print("SAVED TO", f_path)
    original = list(dataset[positional_index][0][0].cpu().numpy())
    original.extend([x for x in range(356, 484)] + [99, 99, 99, 99])
    toks_orig = our_encoding.id_to_event(original)
    # our_encoding.decode_events_to_midi(toks, f_path)
    f_path = os.path.join(args.output_dir, "original{}.mid".format(positional_index))
    print("SAVED TO", f_path)
    our_encoding.decode_events_to_midi(toks_orig, f_path)
    # GENERATION
    model.eval()
    x = torch.squeeze(model(style_embedding, positional_embedding))
    f_path = os.path.join(args.output_dir, "output{}_{}.mid".format(positional_index, style_index))
    print("SAVED TO", f_path)
    output_sequence = torch.argmax(x, dim=-1)
    print(output_sequence.float().std())
    toks_mod = our_encoding.id_to_event(output_sequence.cpu().numpy())
    toks_orig.extend(toks_mod)
    our_encoding.decode_events_to_midi(toks_orig, f_path)
    return





if __name__ == "__main__":
    main()
