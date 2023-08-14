# GeneralizedKWS
[Paper](https://www.isca-speech.org/archive/pdfs/interspeech_2022/r22_interspeech.pdf)

## Setting up the Environment:

```
  git clone https://github.com/Kirandevraj/GeneralizedKWS.git
  cd GeneralizedKWS
  docker build -t gkws-container .
  docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd):/GeneralizedKWS gkws-container
  pip install -r requirements.txt 
```

## Download and Process the data:

```
cd /GeneralizedKWS/preprocess_timit
mkdir TIMIT
```

Download the TIMIT dataset and place it inside the TIMIT folder
```
bash bash_timit_preprocess.sh
```

## Training script

### Download the pretrained model:

```
mkdir /GeneralizedKWS/OpenSeq_pretrained
cd /GeneralizedKWS/OpenSeq_pretrained
```

Download the deepspeech2 model from [openseq2seq](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html) and place it inside the OpenSeq_pretrained folder
```
tar -xf ds2_large.tar.gz
```

### Training

```
mkdir /GeneralizedKWS/training_output
cd /GeneralizedKWS/OpenSeq2Seq_gkws
bash sbatch_timit.sh
```

### Evaluation:

```
mkdir /GeneralizedKWS/testing_output
mkdir /GeneralizedKWS/testing_output/embeddings
cd /GeneralizedKWS/evaluation_codes/test_openseq_0_timit
bash test_folders.sh
```

## Cite as:

```
@inproceedings{r22_interspeech,
  author={Kirandevraj R and Vinod Kumar Kurmi and Vinay Namboodiri and C V Jawahar},
  title={{Generalized Keyword Spotting using ASR embeddings}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={126--130},
  doi={10.21437/Interspeech.2022-10450}
}
```