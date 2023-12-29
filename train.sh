
#TRAIN SCRIPT
python3 train.py \
-lr 0.0001 \
-feature_size 128 \
-input_dir ../dataset_3_genres_1000 \
-output_dir ../condensed_layers \
-batch_size 5 \
-n_workers 1 \
-max_sequence 1000 \
-transition_features 64 \
--rpr

