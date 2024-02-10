

#-continue_epoch 22 \
#-continue_weights ../test_non_linear_positional_embeddings/weights/epoch_0022.pickle \

python3 train_reconstructor.py \
-lr 0.0001 \
-continue_epoch 79 \
-continue_weights ../sanity_test/weights/epoch_0079.pickle \
-input_dir ../dataset_embeddings \
-output_dir ../sanity_test \
-batch_size 4 \
-n_workers 1 \
-max_sequence 1000 \
--rpr


