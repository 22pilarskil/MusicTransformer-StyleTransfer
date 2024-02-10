python3 train_original.py \
-lr 0.0001 \
-continue_epoch 75 \
-continue_weights ../results_original/weights/epoch_0075.pickle \
-input_dir ../dataset_embeddings \
-output_dir ../results_original \
-batch_size 5 \
-n_workers 1 \
-max_sequence 1000 \
--rpr

