#!/bin/sh

input_dir="../dataset_jazz/"
primer_index=3

output_dir="../classical_jazz_6196_results"
weights_dir="${output_dir}/weights/epoch_0009.pickle"
max_sequence=1000

python3 our_generate.py -target_seq_length $max_sequence -midi_root $input_dir -primer_file $primer_index -output_dir $output_dir -model_weights $weights_dir -max_sequence $max_sequence --rpr
