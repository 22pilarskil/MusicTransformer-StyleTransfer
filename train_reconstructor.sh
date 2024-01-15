python3 train_reconstructor.py \
-lr 0.0001 \
-continue_epoch 43 \
-continue_weights ../test_double_masking_varied_length/weights/epoch_0043.pickle \
-input_dir ../dataset_embeddings \
-output_dir ../test_double_masking_varied_length \
-batch_size 5 \
-n_workers 1 \
-max_sequence 1000 \
--rpr


