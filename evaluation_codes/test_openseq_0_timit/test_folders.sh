#!/bin/bash

# This script saves the embeddings from the trained model as .npy files.
# Loads these embeddings to calculate score and saves in output_filename

output_filename="/GeneralizedKWS/testing_output/timit_eval_output_test.txt"
touch $output_filename
echo Analysis > $output_filename
echo $filename

mkdir /GeneralizedKWS/testing_output/embeddings/embeddings1

for test_cases in /GeneralizedKWS/evaluation_codes/test_files/*; do
	echo $test_cases >> $output_filename
	# folder name where the checkpoints are saved
	foldername="/GeneralizedKWS/training_output/logdir"
	echo $foldername >> $output_filename
	python list_checkpoints.py --folder $foldername
	fileItemString=$(cat  "${foldername}/checkpoints_list.txt" |tr "\n" " ")
	fileItemArray=($fileItemString)
	Length=${#fileItemArray[@]}
	for i in $(seq $Length)
	do
		checkpoint=${fileItemArray[$[i-1]]}
		echo $checkpoint >> $output_filename
		config="./example_configs/speech2text/ds2_timit.py"
		sed -i "/specific_checkpoint/c\'specific_checkpoint':\"$checkpoint\"," $config
		# echo $config
		test_files=$(find $test_cases -type f | sort -n)
		echo $test_files
		for test_filename in $test_files; do
			echo $test_filename >> $output_filename
			sed -i "/replace/c\"$test_filename\" #replace" $config
			python run.py --config_file $config --mode eval --num_gpus 1 --batch_size_per_gpu 1 \
				--logdir $foldername
			class="$(($(cat $test_filename | wc -l)-1))"
		done
		cd /GeneralizedKWS/evaluation_codes/multiview/Scripts
		python comparing_timit.py >> $output_filename
		cd /GeneralizedKWS/evaluation_codes/test_openseq_0_timit/
	done
done

echo "results:"

cat $output_filename

echo -----done-----



