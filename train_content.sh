
#TRAIN SCRIPT
python3 train_content.py \
-lr 0.0001 \
-feature_size 128 \
-continue_epoch 8 \
-continue_weights ../results_content_updated/weights/epoch_0008.pickle \
-input_dir ../dataset_separated \
-output_dir ../results_content_updated \
-batch_size 3 \
-n_workers 0 \
-max_sequence 1000 \
-transition_features 64 \
--rpr


