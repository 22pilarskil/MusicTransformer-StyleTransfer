python3 generate_embeddings.py \
-input_dir ../dataset_separated \
-continue_weights ../results_3_genres/weights/epoch_0034.pickle \
-max_sequence 2048 --rpr \
-output_dir ../dataset_separated_embeddings \
-batch_size 1

