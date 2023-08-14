import glob
import subprocess
import librosa
import numpy as np

def fix_audio(file):
	wav,sr = librosa.load(file, sr=16000)
	add_silence = 16000 - len(wav)
	half = int(add_silence/2)
	wav = np.append(np.array([0] * half), wav, axis = 0)
	wav = np.append(wav, np.array([0] * half), axis = 0)
	librosa.output.write_wav(file,y=wav, sr=sr)
	return

def main(split):
	allfiles = glob.glob("/GeneralizedKWS/preprocess_timit/timit3/selected_words_" + split + "/*/*.wav")
	all_length = []
	for pos, item in enumerate(allfiles):
		if pos % 100 == 0:
			print(pos, len(allfiles))
		length = subprocess.check_output(["soxi", "-D", item])
		length = float((length).decode("utf-8").strip())
		if length < 1:
			fix_audio(item)
	return

if __name__ == '__main__':
	main("train")
	main("test")


