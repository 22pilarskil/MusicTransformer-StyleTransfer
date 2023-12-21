#!/bin/sh

input_dir="../classical_jazz_6196_embeddings"
positional_index=$1
style_index=$2

output_dir="../style_transfer_3"
weights_dir="${output_dir}/weights/epoch_0015.pickle"
max_sequence=1000

python3 reconstruct.py -target_seq_length $max_sequence -midi_root $input_dir -style_file $style_index -positional_file $positional_index -output_dir $output_dir -model_weights $weights_dir -max_sequence $max_sequence --rpr
