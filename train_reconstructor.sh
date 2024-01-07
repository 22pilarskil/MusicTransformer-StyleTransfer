python3 train_reconstructor.py \
-lr 0.0001 \
-continue_epoch 13 \
-continue_weights ../results_embeddings/results/epoch_0013.pickle \
-input_dir ../dataset_embeddings \
-output_dir ../test \
-batch_size 5 \
-n_workers 1 \
-max_sequence 1000 \
--rpr


