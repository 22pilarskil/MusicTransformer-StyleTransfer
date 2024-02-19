#!/bin/sh

content_genre=$1
content_index=$2
style_genre=$3
style_index=$4
primer_length=$5
random=$6

echo "THIS IS {$6}"
dir="0022"

output_dir="../test_generate"
#weights_dir="../test_double_masking_varied_length/weights/epoch_0043.pickle"
#weights_dir="../test_high_dropout/weights/epoch_0019.pickle"
#weights_dir="../test_non_linear_positional_embeddings/weights/epoch_${dir}.pickle"
#weights_dir="../test_double_masking/weights/epoch_0039.pickle"
weights_dir="../test_embed_concat_visualized_similarity_penalty/weights/epoch_0056.pickle"
max_sequence=1000

python3 our_generate.py \
-output_dir $output_dir \
-continue_weights $weights_dir \
-max_sequence $max_sequence \
-content_genre $content_genre \
-content_index $content_index \
-style_genre $style_genre \
-style_index $style_index \
-primer_length $primer_length \
-random $random \
-n_layers 6 \
-dropout 0 \
--rpr \
