python3 triplet_tester.py \
-epochs 80 \
-continue_weights ../results_style/weights/epoch_0048.pickle \
--rpr \
-batch_size 1 \
-input_dir ../dataset_3_genres_1000/ \
-max_sequence 1000 \
-feature_size 128
