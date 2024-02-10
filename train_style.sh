
#TRAIN SCRIPT
python3 train_style.py \
-lr 0.0001 \
-feature_size 128 \
-continue_epoch 17 \
-continue_weights ../results_style_updated_vocab_variable_length/weights/epoch_0017.pickle \
-input_dir ../dataset_3_genres_1000 \
-output_dir ../results_style_updated_vocab_variable_length \
-batch_size 5 \
-n_workers 1 \
-max_sequence 1000 \
-transition_features 64 \
--rpr

