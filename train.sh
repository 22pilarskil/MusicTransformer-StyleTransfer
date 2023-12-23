
#TRAIN SCRIPT
python3 train.py \
-lr 0.0001 \
-continue_weights ../feature_size_64/weights/epoch_0009.pickle \
-continue_epoch 11 \
-feature_size 64 \
-input_dir ../dataset \
-output_dir ../feature_size_64 \
-batch_size 5 \
-n_workers 1 \
-max_sequence 1000 \
--rpr

