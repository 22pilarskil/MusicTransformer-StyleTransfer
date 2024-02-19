

#-continue_epoch 22 \
#-continue_weights ../test_non_linear_positional_embeddings/weights/epoch_0022.pickle \

#-continue_epoch 99 \
#-continue_weights ../test_embed_concat/weights/epoch_0099.pickle \

python3 train_reconstructor.py \
-epochs 200 \
-lr 0.0001 \
-input_dir ../dataset_embeddings \
-output_dir ../test_embed_concat_visualized_similarity_penalty \
-batch_size 16 \
-n_workers 1 \
-max_sequence 1000 \
-n_layers 1 \
--rpr


