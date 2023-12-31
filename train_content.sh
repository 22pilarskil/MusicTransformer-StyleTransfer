
#TRAIN SCRIPT
python3 train_content.py \
-lr 0.0001 \
-feature_size 128 \
-input_dir ../dataset_separated \
-output_dir ../results_content_dataset_separated \
-batch_size 1 \
-n_workers 0 \
-max_sequence 1000 \
-transition_features 64 \
--rpr


