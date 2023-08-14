import os
import glob
import argparse
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--folder',type=str, 
		                    help='logdir')

	args = parser.parse_args()
	logdir = args.folder
	print(logdir)
	files = glob.glob(os.path.join(args.folder, "*.meta"))
	# files.sort(key=lambda x: os.path.getctime(x))
	files.sort(key=natural_keys)
	# print(files)
	for pos in range(len(files)):
		files[pos] = files[pos].replace(".meta", "")
	# print(files)
	with open(os.path.join(logdir,"checkpoints_list.txt"), 'w') as f:
	    # for item in files[::-1]:
	    for item in [files[-1]]:
	        f.write("%s\n" % item)
