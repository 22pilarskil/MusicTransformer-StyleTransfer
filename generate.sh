#!/bin/sh

content_genre=$1
content_index=$2
style_genre=$3
style_index=$4
primer_length=$5

dir="0022"

output_dir="../test_generate"
#weights_dir="../test_double_masking_varied_length/weights/epoch_0043.pickle"
#weights_dir="../test_high_dropout/weights/epoch_0019.pickle"
#weights_dir="../test_non_linear_positional_embeddings/weights/epoch_${dir}.pickle"
#weights_dir="../test_double_masking/weights/epoch_0039.pickle"
weights_dir="../sanity_test/weights/epoch_0079.pickle"
max_sequence=1000

python3 our_generate.py \
-batch_size 5 \
-output_dir $output_dir \
-continue_weights $weights_dir \
-max_sequence $max_sequence \
-content_genre $content_genre \
-content_index $content_index \
-style_genre $style_genre \
-style_index $style_index \
-primer_length $primer_length
-dropout 0 \
--rpr \
