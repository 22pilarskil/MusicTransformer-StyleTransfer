import torch
import torch.nn as nn
import os
import random
import pickle

from third_party.midi_processor.processor import decode_midi, encode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from model.reconstructor import Reconstructor
from dataset.e_piano import create_embedding_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.e_piano import create_separated_datasets
from utilities.constants import *
from utilities.device import get_device, use_cuda

import encoding as our_encoding

content_genre = "jazz"
content_index = "0"
style_genre = "jazz"
style_index = "0"
mask = False
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

    train_dataset, test_dataset, val_dataset = create_embedding_datasets("../dataset_embeddings", args.max_sequence)

    val_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)

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

    file_path_content = f"../dataset_3_genres_1000/train/{content_genre}/train-{content_index}.midi.pickle"
    content_midi = torch.Tensor(pickle.load(open(file_path_content, "rb"))).int().to(get_device())
    toks = our_encoding.id_to_event(content_midi.cpu().numpy())
    f_path = os.path.join(args.output_dir, f"output_{content_genre}{content_index}.mid")
    our_encoding.decode_events_to_midi(toks, f_path)

    file_path_style = f"../dataset_3_genres_1000/train/{style_genre}/train-{style_index}.midi.pickle"
    style_midi = torch.Tensor(pickle.load(open(file_path_style, "rb"))).int().to(get_device())
    toks = our_encoding.id_to_event(style_midi.cpu().numpy())
    f_path = os.path.join(args.output_dir, f"output_{style_genre}{style_index}.mid")
    our_encoding.decode_events_to_midi(toks, f_path)
    content_embedding = content_model(content_midi.unsqueeze(dim=0))
    style_embedding = style_model(style_midi.unsqueeze(dim=0))

    x = iter(val_loader)
    batch = next(x)
    
    style_embedding = batch[1]
    content_embedding = batch[2]
    content_midi = batch[0]
    print(content_embedding)
    print(content_midi.shape)
    print(style_embedding.shape)
    print(content_embedding.shape)
    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        rand_seq = model.generate(style_embedding, content_embedding, content_midi, 750, args.target_seq_length)
        # rand_seq = model.generate_one_shot(style_embedding, content_embedding, content_midi, mask)

        print('rL ', rand_seq.shape)
        f_path = os.path.join(args.output_dir, f"style_{style_genre}{style_index}-1content_{content_genre}{content_index}.mid")
        print(rand_seq)
        # decode_midi(rand_seq[0].cpu().numpy(), file_path=f_path)
        toks = our_encoding.id_to_event(rand_seq[0].cpu().numpy())
        our_encoding.decode_events_to_midi(toks, f_path)




if __name__ == "__main__":
    main()
