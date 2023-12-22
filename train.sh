
#TRAIN SCRIPT
python3 train.py \
-continue_epoch 0 \
-continue_weights ../batching_new/weights/epoch_0004.pickle \
-lr 0.0001 \
-feature_size 128 \
-input_dir ../dataset \
-output_dir ../batching_2.0_margin \
-batch_size 5 \
-n_workers 1 \
-max_sequence 1000 \
--rpr

