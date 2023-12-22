
#TRAIN SCRIPT
python3 train.py \
-continue_epoch 1 \
-continue_weights ../batching/weights/epoch_0021.pickle \
-lr 0.0001 \
-feature_size 128 \
-input_dir ../dataset \
-output_dir ../batching_new \
-batch_size 5 \
-n_workers 1 \
-max_sequence 1000 \
--rpr

