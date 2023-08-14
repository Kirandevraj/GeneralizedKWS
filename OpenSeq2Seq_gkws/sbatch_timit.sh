#!/bin/bash

ctc_weight=1.0
triplet_weight=20.0
logdir="/GeneralizedKWS/training_output/logdir"
ngpus=1
config_fp="example_configs/speech2text/ds2_timit.py"

python run.py --config_file $config_fp --mode train \
				--ctc_weight $ctc_weight --triplet_weight $triplet_weight --logdir $logdir \
				--num_gpus $ngpus



