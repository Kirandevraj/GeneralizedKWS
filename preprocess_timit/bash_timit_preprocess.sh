# Create a folder TIMIT in the current directory.
# Extract the data inside the TIMIT folder

python dataprep_prp1s.py
python kws_selection.py
python dataprep_1s.py

# rm -r /data/preprocess_timit/timit3/timit_train_ds
# rm -r /data/preprocess_timit/timit3/timit_test_ds

# move the processed timit3 folder to the parent directory.
mv timit3 ../

# rm -r TIMIT