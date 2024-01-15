#!/bin/sh

output_dir="../test_generate"
weights_dir="../test_double_masking_varied_length/weights/epoch_0043.pickle"
max_sequence=1000

python3 our_generate.py \
-target_seq_length $max_sequence \
-output_dir $output_dir \
-model_weights $weights_dir \
-max_sequence $max_sequence \
-feature_size 128 \
-dropout 0 \
--rpr \
