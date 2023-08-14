import argparse
import glob
import sys
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
import os
import librosa
import numpy as np
import csv
import subprocess

parser = argparse.ArgumentParser(description='File Names')
parser.add_argument('--filename', help='TIMIT_filename', default='/GeneralizedKWS/preprocess_timit/TIMIT/')
parser.add_argument('--store',help='store_filename', default='/GeneralizedKWS/preprocess_timit/TIMIT/timit3/timit_test_ds')
args = parser.parse_args()

stop_words = set(stopwords.words('english')) 
if os.path.exists(args.store) == False:
    os.makedirs("/GeneralizedKWS/preprocess_timit/timit3", exist_ok=True)
    os.makedirs("/GeneralizedKWS/preprocess_timit/timit3/timit_train_ds", exist_ok=True)
    os.makedirs("/GeneralizedKWS/preprocess_timit/timit3/timit_test_ds", exist_ok=True)

def main(split):
    if split == "train":
        args.store = "/GeneralizedKWS/preprocess_timit/timit3/timit_train_ds"
        word_files = glob.glob(args.filename + "data/lisa/data/timit/raw/TIMIT/TRAIN/*/*/*.WRD")
    elif split == "test":
        args.store = "/GeneralizedKWS/preprocess_timit/timit3/timit_test_ds"
        word_files = glob.glob(args.filename + "data/lisa/data/timit/raw/TIMIT/TEST/*/*/*.WRD")
    else:
        print("ERROR!")
        exit(0)
    files_audio = []
    words = {}
    less_than_1s_count = 0
    for iteration, file in enumerate(word_files):
        f = open(file,'r')
        d = f.read()
        lines = d.split('\n')
        wav,sr = librosa.load(file[:-3] + "WAV", sr=16000)
        for item in lines[:-1]:
            sample_pos = item.split(" ")
            if len(sample_pos[-1]) > 3:
                if sample_pos[-1] not in stop_words:
                    audio_sample = int(sample_pos[1]) - int(sample_pos[0])
                    if audio_sample < 16000:
                        append_sample = 16000 - audio_sample
                        offset = int(append_sample/2)
                        if int(sample_pos[0]) - offset >= 0:
                            sample_pos[0] = str(int(sample_pos[0]) - offset)
                        elif (int(sample_pos[0]) - offset) < 0:
                            sample_pos[0] = 0
                            less_than_1s_count += 1
                        length = subprocess.check_output(["soxi", "-D", file[:-3] + "WAV"])
                        if float(length) >= (int(sample_pos[1]) + offset)/16000:
                            sample_pos[1] = str(int(sample_pos[1]) + offset)
                        elif float(length) < (int(sample_pos[1]) + offset)/16000:
                            sample_pos[1] = str(int(sample_pos[1]) + offset)
                            less_than_1s_count += 1

                    if sample_pos[-1] in words:
                        words[sample_pos[-1]] += 1
                        librosa.output.write_wav(args.store + '/' + str(sample_pos[-1]) +'/'+ str(words[sample_pos[-1]]+1) + '.wav',y=wav[int(sample_pos[0]):int(sample_pos[1])], sr=sr)
                        files_audio.append([file.replace(".WRD", ".wav"), args.store + '/' + str(sample_pos[-1]) +'/'+ str(words[sample_pos[-1]]+1) + '.wav'])
                    else:
                        words[sample_pos[-1]] = 0
                        os.makedirs(args.store + '/' + str(sample_pos[-1]))
                        librosa.output.write_wav(args.store + '/' + str(sample_pos[-1]) +'/'+ str(words[sample_pos[-1]]+1) + '.wav',y=wav[int(sample_pos[0]):int(sample_pos[1])], sr=sr)
                        files_audio.append([file.replace(".WRD", ".wav"), args.store + '/' + str(sample_pos[-1]) +'/'+ str(words[sample_pos[-1]]+1) + '.wav'])
        if iteration % 200 == 0:
            print("pos: ", iteration, len(word_files))
    print("Less than 1s count: ", less_than_1s_count)


if __name__=="__main__":
    main("train")
    main("test")