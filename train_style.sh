
#TRAIN SCRIPT
python3 train_style.py \
-lr 0.0001 \
-feature_size 128 \
-input_dir ../dataset_3_genres_1000 \
-output_dir ../results_3_genres_benchmark_new_masking \
-batch_size 5 \
-n_workers 1 \
-max_sequence 1000 \
-transition_features 64 \
--rpr

