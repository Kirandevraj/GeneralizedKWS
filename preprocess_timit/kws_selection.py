import glob
from nltk.corpus import stopwords 
import os
import random
import sklearn.metrics
from shutil import copyfile
import csv

def main(split):
	kws = glob.glob("/GeneralizedKWS/preprocess_timit/timit3/timit_" + split + "_ds/*")
	kwords = []
	for i in kws:
		kwords.append(i.split('/')[-1])
	allowed_set = []
	allowed_set_occurance = []
	stop_words = set(stopwords.words('english')) 
	words_with_sp_char = []
	words_with_sp_char_count = 0
	for i in kwords:
		if "'" in i:
			words_with_sp_char.append(i)
			words_with_sp_char_count += len(glob.glob("/GeneralizedKWS/preprocess_timit/timit3/timit_" + split + "_ds/"+i+"/*"))
		if split == "test":
			if len(glob.glob("/GeneralizedKWS/preprocess_timit/timit3/timit_" + split + "_ds/"+i+"/*")) >= 2 and "'" not in i:
				allowed_set.append(i)
				allowed_set_occurance.append(len(glob.glob("/GeneralizedKWS/preprocess_timit/timit3/timit_" + split + "_ds/"+i+"/*")))
		elif split == "train":
			if len(glob.glob("/GeneralizedKWS/preprocess_timit/timit3/timit_" + split + "_ds/"+i+"/*")) >= 2 and "'" not in i:
				allowed_set.append(i)
				allowed_set_occurance.append(len(glob.glob("/GeneralizedKWS/preprocess_timit/timit3/timit_" + split + "_ds/"+i+"/*")))
		else:
			print("ERROR")
	print("words with sp chars: ", words_with_sp_char, words_with_sp_char_count)
	yx = zip(allowed_set_occurance,allowed_set)
	yx = sorted(yx)[::-1]
	x_sorted = [x for y, x in list(yx)]
	final = x_sorted[:] #change 1
	print(len(final))
	files_to_files = []
	for i in final:
		if len(str(i)) > 4:
			os.makedirs('/GeneralizedKWS/preprocess_timit/timit3/selected_words_' + split + '/' + i)
			files = glob.glob("/GeneralizedKWS/preprocess_timit/timit3/timit_" + split + "_ds/" + i + "/*")
			for item in files:
				copyfile(item, "/GeneralizedKWS/preprocess_timit/timit3/selected_words_" + split + "/" + i + "/" +item.split("/")[-1])
				files_to_files.append([item, "/GeneralizedKWS/preprocess_timit/timit3/selected_words_" + split + "/" + i + "/" +item.split("/")[-1]])
	return

if __name__ == "__main__":
	main("train")
	main("test")
