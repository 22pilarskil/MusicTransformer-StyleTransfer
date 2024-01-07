import torch
import torch.nn as nn
import os
import random
import pickle

from third_party.midi_processor.processor import decode_midi, encode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from model.reconstructor import Reconstructor
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi
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

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    style_model_path = "../results_style/weights/epoch_0048.pickle"
    content_model_path = "../results_content/weights/epoch_0052.pickle"

    style_model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                feature_size=args.feature_size, max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    content_model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                feature_size=args.feature_size, max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    style_model.load_state_dict(torch.load(style_model_path))
    content_model.load_state_dict(torch.load(content_model_path))

    model = Reconstructor(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

    print(get_device())
    model.load_state_dict(torch.load(args.model_weights, map_location=torch.device('cpu')))

    file_path_content = "../dataset_3_genres_1000/val/classical/val-0.midi.pickle"
    content_midi = torch.Tensor(pickle.load(open(file_path_content, "rb"))).int().to(get_device())
    toks = our_encoding.id_to_event(content_midi.cpu().numpy())
    f_path = os.path.join(args.output_dir, "output_content.mid")
    our_encoding.decode_events_to_midi(toks, f_path)

    file_path_style = "../dataset_3_genres_1000/val/classical/val-0.midi.pickle"
    style_midi = torch.Tensor(pickle.load(open(file_path_style, "rb"))).int().to(get_device())
    toks = our_encoding.id_to_event(style_midi.cpu().numpy())
    f_path = os.path.join(args.output_dir, "output_style.mid")
    our_encoding.decode_events_to_midi(toks, f_path)

    content_embedding = content_model(content_midi.unsqueeze(dim=0))
    style_embedding = style_model(style_midi.unsqueeze(dim=0))
    print(style_embedding.shape)

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        rand_seq = model.generate(style_embedding, content_embedding, args.target_seq_length)
        print('rL ', rand_seq.shape)
        f_path = os.path.join(args.output_dir, "output.mid")
        print(rand_seq)
        # decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)
        toks = our_encoding.id_to_event(rand_seq[0].cpu().numpy())
        our_encoding.decode_events_to_midi(toks, f_path)




if __name__ == "__main__":
    main()
