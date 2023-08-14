# Copyright (c) 2018 NVIDIA Corporation
"""Data Layer for Speech-to-Text models"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
import six
import math
import librosa
from six import string_types
from six.moves import range

from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary
from .speech_utils import get_speech_features_from_file, get_speech_features
import sentencepiece as spm

import random
import os
from tensorflow.python.platform import gfile
import re
import hashlib
import math
from tensorflow.python.util import compat
from numpy.lib.stride_tricks import as_strided

import sys
from glob import glob

RANDOM_SEED = 0
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

np.random.seed(RANDOM_SEED)

# numpy.fft MKL bug: https://github.com/IntelPython/mkl_fft/issues/11
if hasattr(np.fft, 'restore_all'):
  np.fft.restore_all()

class Speech2TextDataLayer(DataLayer):
  """Speech-to-text data layer class."""
  @staticmethod
  def get_required_params():
    return dict(DataLayer.get_required_params(), **{
        'num_audio_features': int,
        'input_type': ['spectrogram', 'mfcc', 'logfbank'],
        'vocab_file': str,
        'dataset_files': list,
    })

  @staticmethod
  def get_optional_params():
    return dict(DataLayer.get_optional_params(), **{
        'backend': ['psf', 'librosa'],
        'augmentation': dict,
        'pad_to': int,
        'max_duration': float,
        'min_duration': float,
        'bpe': bool,
        'autoregressive': bool,
        'syn_enable': bool,
        'syn_subdirs': list,
        'window_size': float,
        'window_stride': float,
        'dither': float,
        'norm_per_feature': bool,
        'window': ['hanning', 'hamming', 'none'],
        'num_fft': int,
        'precompute_mel_basis': bool,
        'sample_freq': int,
        'gain': float,
        'features_mean': np.ndarray,
        'features_std_dev': np.ndarray,
        'dataset': str
    })

  def __init__(self, params, model, num_workers, worker_id):
    """Speech-to-text data layer constructor.
    See parent class for arguments description.
    Config parameters:
    * **backend** (str) --- audio pre-processing backend
      ('psf' [default] or librosa [recommended]).
    * **num_audio_features** (int) --- number of audio features to extract.
    * **input_type** (str) --- could be either "spectrogram" or "mfcc".
    * **vocab_file** (str) --- path to vocabulary file or sentencepiece model.
    * **dataset_files** (list) --- list with paths to all dataset .csv files.
    * **augmentation** (dict) --- optional dictionary with data augmentation
      parameters. Can contain "speed_perturbation_ratio", "noise_level_min" and
      "noise_level_max" parameters, e.g.::
        {
          'speed_perturbation_ratio': 0.05,
          'noise_level_min': -90,
          'noise_level_max': -60,
        }
      For additional details on these parameters see
      :func:`data.speech2text.speech_utils.augment_audio_signal` function.
    * **pad_to** (int) --- align audio sequence length to pad_to value.
    * **max_duration** (float) --- drop all samples longer than
      **max_duration** (seconds)
    * **min_duration** (float) --- drop all samples shorter than
      **min_duration** (seconds)
    * **bpe** (bool) --- use BPE encodings
    * **autoregressive** (bool) --- boolean indicating whether the model is
      autoregressive.
    * **syn_enable** (bool) --- boolean indicating whether the model is
      using synthetic data.
    * **syn_subdirs** (list) --- must be defined if using synthetic mode.
      Contains a list of subdirectories that hold the synthetica wav files.
    * **window_size** (float) --- window's duration (in seconds)
    * **window_stride** (float) --- window's stride (in seconds)
    * **dither** (float) --- weight of Gaussian noise to apply to input signal
      for dithering/preventing quantization noise
    * **num_fft** (int) --- size of fft window to use if features require fft,
          defaults to smallest power of 2 larger than window size
    * **norm_per_feature** (bool) --- if True, the output features will be
      normalized (whitened) individually. if False, a global mean/std over all
      features will be used for normalization.
    * **window** (str) --- window function to apply before FFT
      ('hanning', 'hamming', 'none')
    * **num_fft** (int) --- optional FFT size
    * **precompute_mel_basis** (bool) --- compute and store mel basis. If False,
      it will compute it for every get_speech_features call. Default: False
    * **sample_freq** (int) --- required for precompute_mel_basis
    """
    super(Speech2TextDataLayer, self).__init__(params, model,
                                               num_workers, worker_id)
    # self.data_dir = "/ssd_scratch/cvit/kiran/speech_commands"
    # self.data_index = {"train": [],
    #                    "test": [],
    #                    "validation": []}
    # silence_percentage = 0.0
    # unknown_percentage = 0.0
    # wanted_words = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
    # validation_percentage = 10
    # testing_percentage = 10

    # self.prepare_data_index(silence_percentage=silence_percentage,
    #                         unknown_percentage=unknown_percentage,
    #                         wanted_words=wanted_words,
    #                         validation_percentage=validation_percentage,
    #                         testing_percentage=testing_percentage)

    self.params['autoregressive'] = self.params.get('autoregressive', False)
    self.autoregressive = self.params['autoregressive']
    self.params['bpe'] = self.params.get('bpe', False)
    if self.params['bpe']:
      self.sp = spm.SentencePieceProcessor()
      self.sp.Load(self.params['vocab_file'])
      self.params['tgt_vocab_size'] = len(self.sp) + 1
    else:
      self.params['char2idx'] = load_pre_existing_vocabulary(
          self.params['vocab_file'], read_chars=True,
      )
      if not self.autoregressive:
        # add one for implied blank token
        self.params['tgt_vocab_size'] = len(self.params['char2idx']) + 1
      else:
        num_chars_orig = len(self.params['char2idx'])
        self.params['tgt_vocab_size'] = num_chars_orig + 2
        self.start_index = num_chars_orig
        self.end_index = num_chars_orig + 1
        self.params['char2idx']['<S>'] = self.start_index
        self.params['char2idx']['</S>'] = self.end_index
        self.target_pad_value = self.end_index
      self.params['idx2char'] = {i: w for w,
                                 i in self.params['char2idx'].items()}
    self.target_pad_value = 0

    self._files = None
    if self.params["interactive"]:
      return
    for csv in params['dataset_files']:
      files = pd.read_csv(csv, encoding='utf-8')
      if self._files is None:
        self._files = files
      else:
        self._files = self._files.append(files)

    if self.params['mode'] != 'infer':
      cols = ['wav_filename', 'transcript']
    else:
      cols = 'wav_filename'

    self.all_files = self._files.loc[:, cols].values
    self._files = self.split_data(self.all_files)

    self._size = self.get_size_in_samples()
    self._dataset = None
    self._iterator = None
    self._input_tensors = None

    self.testing_var = None

    self.params['min_duration'] = self.params.get('min_duration', -1.0)
    self.params['max_duration'] = self.params.get('max_duration', -1.0)
    self.params['window_size'] = self.params.get('window_size', 20e-3)
    self.params['window_stride'] = self.params.get('window_stride', 10e-3)
    self.params['sample_freq'] = self.params.get('sample_freq', 16000)

    self.params['dataset'] = self.params.get('dataset', 'timit')

    if self.params['dataset'] == "timit":
      kws = sorted(os.listdir("/GeneralizedKWS/timit3/selected_words_train"))
      labels_dictionary = {}
      for pos, item in enumerate(kws):
        labels_dictionary[item] = pos
      self.params['labels'] = labels_dictionary

    elif self.params['dataset'] == "nl":
      kws = sorted(os.listdir("/ssd_scratch/cvit/kiran/nl_165/train_split"))
      labels_dictionary = {}
      for pos, item in enumerate(kws):
        labels_dictionary[item] = pos
      self.params['labels'] = labels_dictionary

    elif self.params['dataset'] == "mono_en":
      kws = sorted(os.listdir("/ssd_scratch/cvit/kiran/en/train_split"))
      labels_dictionary = {}
      for pos, item in enumerate(kws):
        labels_dictionary[item] = pos
      self.params['labels'] = labels_dictionary

    elif self.params['dataset'] == "google":
      self.params['labels'] = ['no', 'happy', 'one', 'right', 'cat', 'tree', 'bed', 'stop', 'on', 'nine', \
                            'sheila', 'house', 'yes', 'up', 'eight', 'seven', 'off', 'visual', 'marvin', \
                            'four', 'two', 'zero', 'six', 'down', 'bird', 'go', 'wow', 'learn', 'dog', \
                            'backward', 'forward', 'follow', 'five', 'left', 'three']
    elif self.params["dataset"] == "mswc":
      self.params['labels'] = {'rw': {'uwo': 0, 'kuri': 1, 'kuko': 2, 'ndetse': 3, 'agira': 4, 'imbere': 5, 'kuba': 6, 'yavuze': 7, 'kubera': 8, 'yari': 9, 'cyane': 10, 'abantu': 11, 'gihe': 12, 'ntabwo': 13, 'indirimbo': 14, 'bari': 15, 'neza': 16, 'ngo': 17, 'aba': 18, 'bamwe': 19, 'gusa': 20, 'nta': 21, 'ibi': 22, 'ubu': 23, 'umwe': 24, 'imana': 25, 'ari': 26, 'kandi': 27, 'bose': 28, 'afite': 29, 'iki': 30, 'rwanda': 31, 'umuntu': 32, 'uko': 33, 'munsi': 34, 'bwa': 35, 'aho': 36, 'cya': 37, 'kugira': 38, 'cyangwa': 39, 'ibyo': 40, 'icyo': 41, 'iyo': 42, 'muri': 43, 'benshi': 44, 'gukora': 45, 'byo': 46, 'perezida': 47, 'uyu': 48, 'iyi': 49, 'mbere': 50, 'yagize': 51, 'buri': 52, 'hari': 53, 'abana': 54, 'cyo': 55, 'ariko': 56, 'igihe': 57, 'ubwo': 58, 'byose': 59, 'avuga': 60, 'nyuma': 61, 'ati': 62, 'iri': 63, 'ibintu': 64, 'leta': 65, 'buryo': 66, 'kagame': 67}, 'nl': {'hij': 68, 'was': 69, 'voor': 70, 'zijn': 71, 'met': 72, 'niet': 73, 'dat': 74}, 'de': {'bei': 75, 'alle': 76, 'zeit': 77, 'ihrer': 78, 'bitte': 79, 'gut': 80, 'konnte': 81, 'macht': 82, 'wäre': 83, 'noch': 84, 'hast': 85, 'seiner': 86, 'einer': 87, 'wurde': 88, 'wurden': 89, 'zum': 90, 'dies': 91, 'hier': 92, 'dir': 93, 'mal': 94, 'einige': 95, 'können': 96, 'auch': 97, 'herr': 98, 'ihnen': 99, 'jedoch': 100, 'heute': 101, 'eines': 102, 'lassen': 103, 'hatte': 104, 'werden': 105, 'habe': 106, 'liegt': 107, 'nach': 108, 'mehr': 109, 'einmal': 110, 'kein': 111, 'selbst': 112, 'wieder': 113, 'doch': 114, 'immer': 115, 'meine': 116, 'über': 117, 'sich': 118, 'wenn': 119, 'später': 120, 'müssen': 121, 'haben': 122, 'denn': 123, 'für': 124, 'geht': 125, 'andere': 126, 'unter': 127, 'dazu': 128, 'seine': 129, 'muss': 130, 'seinen': 131, 'beim': 132, 'vom': 133, 'soll': 134, 'zwischen': 135, 'gegen': 136, 'dort': 137, 'nur': 138, 'waren': 139, 'sollte': 140, 'mit': 141, 'dieses': 142, 'von': 143, 'dann': 144, 'den': 145, 'möchte': 146, 'will': 147, 'fünf': 148, 'durch': 149, 'dabei': 150, 'was': 151, 'ihre': 152, 'eine': 153, 'einen': 154, 'des': 155, 'wer': 156, 'gibt': 157, 'einem': 158, 'ich': 159, 'alles': 160, 'vier': 161, 'stadt': 162, 'wie': 163, 'drei': 164, 'ersten': 165, 'dieser': 166, 'teil': 167, 'diese': 168, 'diesen': 169, 'ihm': 170, 'aber': 171, 'jetzt': 172, 'nichts': 173, 'ein': 174, 'man': 175, 'nicht': 176, 'seinem': 177, 'machen': 178, 'ihren': 179, 'auf': 180, 'ihn': 181, 'wird': 182, 'ganz': 183, 'bereits': 184, 'sein': 185, 'diesem': 186, 'steht': 187, 'erst': 188, 'viele': 189, 'kann': 190, 'sind': 191, 'sehr': 192, 'schon': 193, 'nun': 194, 'einfach': 195, 'zwei': 196, 'ins': 197, 'oder': 198, 'vor': 199, 'wir': 200, 'mich': 201, 'als': 202, 'hauptstadt': 203, 'keine': 204, 'dass': 205, 'heißt': 206, 'kommt': 207, 'hat': 208, 'bis': 209, 'also': 210, 'ihr': 211, 'frau': 212, 'mein': 213, 'lange': 214, 'dem': 215, 'war': 216, 'aus': 217, 'etwas': 218, 'gerade': 219, 'mir': 220, 'anderen': 221, 'ohne': 222, 'viel': 223, 'damit': 224, 'zur': 225, 'uns': 226}, 'ca': {'anys': 227, 'ciutat': 228, 'tres': 229, 'mil': 230, 'diverses': 231, 'estat': 232, 'només': 233, 'sobre': 234, 'tant': 235, 'part': 236, 'tots': 237, 'quatre': 238, 'què': 239, 'anem': 240, 'cas': 241, 'aquestes': 242, 'tot': 243, 'això': 244, 'sud': 245, 'altres': 246, 'havia': 247, 'pel': 248, 'van': 249, 'cada': 250, 'seves': 251, 'lloc': 252, 'fins': 253, 'd’una': 254, 'fer': 255, 'tenen': 256, 'diferents': 257, 'seu': 258, 'després': 259, 'nom': 260, 'aquest': 261, 'fou': 262, 'nord': 263, 'entre': 264, 'així': 265, 'cal': 266, 'molt': 267, 'cinc': 268, 'primer': 269, 'des': 270, 'actualment': 271, 'quan': 272, 'seva': 273, 'però': 274, 'seus': 275, 'venim': 276, 'dia': 277, 'encara': 278, 'durant': 279, 'poden': 280, 'd’un': 281, 'dues': 282, 'són': 283, 's’ha': 284, 'està': 285, 'aquesta': 286, 'forma': 287, 'dos': 288, 'als': 289, 'cap': 290, 'gran': 291, 'poc': 292, 'temps': 293, 'han': 294, 'aquests': 295, 'sant': 296, 'planta': 297, 'troba': 298, 'era': 299, 'mateix': 300, 'qui': 301, 'sense': 302, 'pot': 303, 'ser': 304, 'nou': 305, 'casa': 306}, 'en': {'always': 307, 'over': 308, 'party': 309, 'building': 310, 'friend': 311, 'very': 312, 'played': 313, 'during': 314, 'give': 315, 'one': 316, 'made': 317, 'people': 318, 'anything': 319, 'decided': 320, 'who': 321, 'first': 322, 'later': 323, 'without': 324, 'appeared': 325, 'good': 326, 'does': 327, 'language': 328, 'work': 329, 'children': 330, 'green': 331, 'use': 332, 'now': 333, 'built': 334, 'family': 335, 'different': 336, 'again': 337, 'continued': 338, 'night': 339, 'part': 340, 'other': 341, 'day': 342, 'small': 343, 'year': 344, 'time': 345, 'company': 346, 'based': 347, 'could': 348, 'school': 349, 'tell': 350, 'heart': 351, 'things': 352, 'look': 353, 'every': 354, 'well': 355, 'sun': 356, 'west': 357, 'say': 358, 'began': 359, 'road': 360, 'once': 361, 'team': 362, 'would': 363, 'water': 364, 'under': 365, 'light': 366, 'going': 367, 'university': 368, 'local': 369, 'get': 370, 'see': 371, 'located': 372, 'which': 373, 'right': 374, 'because': 375, 'earth': 376, 'might': 377, 'after': 378, 'into': 379, 'gold': 380, 'morning': 381, 'days': 382, 'seen': 383, 'put': 384, 'group': 385, 'saw': 386, 'east': 387, 'long': 388, 'most': 389, 'near': 390, 'boy': 391, 'both': 392, 'album': 393, 'alchemist': 394, 'early': 395, 'high': 396, 'need': 397, 'found': 398, 'here': 399, 'often': 400, 'large': 401, 'asked': 402, 'way': 403, 'still': 404, 'when': 405, 'same': 406, 'next': 407, 'years': 408, 'know': 409, 'between': 410, 'little': 411, 'thing': 412, 'born': 413, 'should': 414, 'each': 415, 'nine': 416, 'some': 417, 'never': 418, 'book': 419, 'known': 420, 'street': 421, 'county': 422, 'mean': 423, 'great': 424, 'name': 425, 'hand': 426, 'station': 427, 'north': 428, 'main': 429, 'district': 430, 'best': 431, 'these': 432, 'read': 433, 'will': 434, 'united': 435, 'house': 436, 'even': 437, 'show': 438, 'where': 439, 'yes': 440, 'city': 441, 'black': 442, 'state': 443, 'make': 444, 'how': 445, 'told': 446, 'everyone': 447, 'done': 448, 'today': 449, 'park': 450, 'went': 451, 'them': 452, 'money': 453, 'woman': 454, 'music': 455, 'sheep': 456, 'end': 457, 'live': 458, 'englishman': 459, 'about': 460, 'sound': 461, 'much': 462, 'eight': 463, 'since': 464, 'like': 465, 'seven': 466, 'set': 467, 'more': 468, 'include': 469, 'mind': 470, 'knew': 471, 'already': 472, 'college': 473, 'written': 474, 'away': 475, 'help': 476, 'served': 477, 'film': 478, 'air': 479, 'desert': 480, 'few': 481, 'number': 482, 'man': 483, 'playing': 484, 'off': 485, 'such': 486, 'old': 487, 'answered': 488, 'red': 489, 'everything': 490, 'ever': 491, 'eyes': 492, 'own': 493, 'several': 494, 'want': 495, 'four': 496, 'those': 497, 'took': 498, 'love': 499, 'play': 500, 'national': 501, 'point': 502, 'lot': 503, 'young': 504, 'matter': 505, 'however': 506, 'blue': 507, 'system': 508, 'town': 509, 'looking': 510, 'been': 511, 'became': 512, 'called': 513, 'nothing': 514, 'must': 515, 'six': 516, 'only': 517, 'game': 518, 'than': 519, 'also': 520, 'before': 521, 'idea': 522, 'through': 523, 'world': 524, 'used': 525, 'another': 526, 'down': 527, 'king': 528, 'named': 529, 'second': 530, 'wanted': 531, 'girl': 532, 'life': 533, 'father': 534, 'many': 535, 'fire': 536, 'times': 537, 'really': 538, 'just': 539, 'around': 540, 'something': 541, 'become': 542, 'south': 543, 'being': 544, 'public': 545, 'better': 546, 'think': 547, 'war': 548, 'last': 549, 'area': 550, 'five': 551, 'wind': 552, 'place': 553, 'home': 554, 'looked': 555, 'back': 556, 'son': 557, 'thought': 558, 'find': 559, 'new': 560, 'men': 561, 'left': 562, 'heard': 563, 'river': 564, 'please': 565, 'office': 566, 'three': 567, 'take': 568, 'then': 569}, 'fr': {'porte': 570, 'grand': 571, 'plusieurs': 572, 'mille': 573, 'cinquante': 574, 'france': 575, 'leur': 576, 'avec': 577, 'partie': 578, 'chemin': 579, 'ont': 580, 'donc': 581, 'sept': 582, 'fils': 583, 'mais': 584, 'groupe': 585, 'sud': 586, 'fut': 587, 'peu': 588, 'non': 589, 'était': 590, 'après': 591, 'même': 592, 'huit': 593, 'encore': 594, 'cinq': 595, 'c’est': 596, 'avait': 597, 'peut': 598, 'ainsi': 599, 'aussi': 600, 'ses': 601, 'vingt': 602, 'fois': 603, 'soutenir': 604, 'nom': 605, 'nord': 606, 'parole': 607, 'première': 608, 'l’amendement': 609, 'entre': 610, 'alors': 611, 'quelques': 612, 'comme': 613, 'tout': 614, 'premier': 615, 'lui': 616, 'madame': 617, 'monsieur': 618, 'été': 619, 'très': 620, 'saint': 621, 'cela': 622, 'oui': 623, 'ensuite': 624, 'depuis': 625, 'd’un': 626, 'ces': 627, 'ville': 628, 'paris': 629, 'bien': 630, 'ils': 631, 'jean': 632, 'père': 633, 'lieu': 634, 'votre': 635, 'cents': 636, 'sous': 637, 'cependant': 638, 'zéro': 639, 'cet': 640, 'être': 641, 'dit': 642, 'trente': 643, 'route': 644, 'elles': 645, 'mon': 646, 'neuf': 647, 'tous': 648, 'trouve': 649, 'd’une': 650, 'six': 651, 'moi': 652, 'puis': 653, 'temps': 654, 'dix': 655, 'trois': 656, 'numéro': 657, 'nous': 658, 'leurs': 659, 'qu’il': 660, 'quarante': 661, 'famille': 662, 'suis': 663, 'qui': 664, 'sans': 665, 'aux': 666, 'ans': 667, 'place': 668, 'faire': 669, 'soixante': 670, 'commune': 671, 'fait': 672, 'également': 673}, 'it': {'sono': 674, 'questo': 675, 'una': 676, 'nella': 677, 'alla': 678, 'dei': 679, 'nel': 680, 'con': 681, 'più': 682, 'sua': 683, 'delle': 684, 'della': 685, 'come': 686, 'gli': 687, 'era': 688, 'anche': 689, 'due': 690}, 'es': {'tres': 691, 'cinco': 692, 'sobre': 693, 'ciudad': 694, 'durante': 695, 'más': 696, 'este': 697, 'está': 698, 'además': 699, 'desde': 700, 'años': 701, 'entre': 702, 'parte': 703, 'tiene': 704, 'hay': 705, 'encuentra': 706, 'para': 707, 'muy': 708, 'fueron': 709, 'esta': 710, 'dos': 711, 'cuatro': 712, 'sus': 713, 'sin': 714, 'pero': 715, 'era': 716, 'como': 717, 'uno': 718, 'son': 719, 'seis': 720, 'ser': 721}}

    elif self.params["dataset"] == "tamil":
      kws = sorted(os.listdir("/ssd_scratch/cvit/kiran/tamil/train_split"))
      labels_dictionary = {}
      for pos, item in enumerate(kws):
        labels_dictionary[item] = pos
      self.params['labels'] = labels_dictionary

    elif self.params["dataset"] == "vallander":
      kws = sorted(os.listdir("/ssd_scratch/cvit/kiran/vallander/train_split"))
      labels_dictionary = {}
      for pos, item in enumerate(kws):
        labels_dictionary[item] = pos
      self.params['labels'] = labels_dictionary

    print("self.params: ", self.params)
    
  

    mel_basis = None
    if (self.params.get("precompute_mel_basis", False) and
        self.params["input_type"] == "logfbank"):
      num_fft = (
          self.params.get("num_fft", None) or
          2**math.ceil(math.log2(
              self.params['window_size']*self.params['sample_freq'])
          )
      )
      mel_basis = librosa.filters.mel(
          self.params['sample_freq'],
          num_fft,
          n_mels=self.params['num_audio_features'],
          fmin=0,
          fmax=int(self.params['sample_freq']/2)
      )
    self.params['mel_basis'] = mel_basis

    if 'n_freq_mask' in self.params.get('augmentation', {}):
      width_freq_mask = self.params['augmentation'].get('width_freq_mask', 10)
      if width_freq_mask > self.params['num_audio_features']:
        raise ValueError(
            "'width_freq_mask'={} should be smaller ".format(width_freq_mask)+
            "than 'num_audio_features'={}".format(
               self.params['num_audio_features']
            )
        )

    if 'time_stretch_ratio' in self.params.get('augmentation', {}):
      print("WARNING: Please update time_stretch_ratio to speed_perturbation_ratio")
      self.params['augmentation']['speed_perturbation_ratio'] = self.params['augmentation']['time_stretch_ratio']

  def prepare_data_index(self, silence_percentage, unknown_percentage,
                           wanted_words, validation_percentage,
                           testing_percentage):
        random.seed(RANDOM_SEED)
        wanted_words_index = {}
        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in gfile.Glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case, since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = which_set(wav_path, validation_percentage, testing_percentage)
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in wanted_words_index:
                self.data_index[set_index].append({'label': word, 'file': wav_path})
            else:
                unknown_index[set_index].append({'label': word, 'file': wav_path})
        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for index, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word +
                                ' in labels but only found ' +
                                ', '.join(all_words.keys()))
        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path
                })
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

  def get_data(self, how_many, offset, mode):
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        data = []
        seq_lens = []
        sample_files = []
        labels = np.zeros(sample_count)
        pick_deterministically = (mode != 'training')
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]
            sample_files.append(sample['file'])

        return sample_files

  def get_duplicates(self, labels, offset, mode):
        candidates = self.data_index[mode]
        how_many = len(labels)
        duplicate_labels = np.zeros(len(labels))
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        data = []
        seq_lens = []
        pick_deterministically = False
        sample_files = []
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                while True:
                    sample_index = np.random.randint(len(candidates))
                    if self.word_to_index[candidates[sample_index]['label']] == labels[i]:
                        break
            sample = candidates[sample_index]
            sample_file.append(sample['file'])
        return sample_files

  def get_nonduplicates(self, labels, offset, mode):
        candidates = self.data_index[mode]
        how_many = len(labels)
        nonduplicate_labels = np.zeros(len(labels))
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        data = []
        seq_lens = []
        pick_deterministically = False
        sample_files = []
        for i in xrange(offset, offset + sample_count):
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                while True:
                    sample_index = np.random.randint(len(candidates))
                    if self.word_to_index[candidates[sample_index]['label']] != labels[i]:
                        break
            sample = candidates[sample_index]
            sample_file.append(sample['file'])
        return sample_files

  def split_data(self, data):
    if self.params['mode'] != 'train' and self._num_workers is not None:
      size = len(data)
      start = size // self._num_workers * self._worker_id
      if self._worker_id == self._num_workers - 1:
        end = size
      else:
        end = size // self._num_workers * (self._worker_id + 1)
      return data[start:end]
    else:
      return data

  @property
  def iterator(self):
    """Underlying tf.data iterator."""
    return self._iterator

  def build_graph(self):
    with tf.device('/cpu:0'):

      """Builds data processing graph using ``tf.data`` API."""
      if self.params['mode'] != 'infer':
        # self._dataset = tf.data.Dataset.from_tensor_slices(self._files)

        # if self.params['shuffle']:
        #   self._dataset = self._dataset.shuffle(self._size)

        def get_files(dir_path, label):
          globbed = tf.string_join([dir_path, '*.wav'])
          files = tf.matching_files(globbed)
          return tf.data.Dataset.from_tensor_slices(files)

        if self.params["dataset"] == "google":
          directory = "/ssd_scratch/cvit/kiran/speech_commands_with_splits/train_split"
          classes = sorted(glob(directory + '/*/')) # final slash selects directories only
          num_classes = len(classes)
          print("Number of classes: ", num_classes)
          labels = classes
          dirs = tf.data.Dataset.from_tensor_slices((classes, labels))               # 1
      

        if self.params["dataset"]=="mswc":
            # print("multilingual: ")
            directory = "/ssd_scratch/cvit/kiran/selected_mswc/mswc_selected_train/*"
            classes = sorted(glob(directory + '/*/')) # final slash selects directories only
            num_classes = len(classes)
            print("Number of classes: ", num_classes)
            labels = classes
            dirs = tf.data.Dataset.from_tensor_slices((classes, labels))               # 1
        
        elif self.params["dataset"]=="nl":
            # print("Monolingual:")
            directory = "/ssd_scratch/cvit/kiran/nl_165/train_split"
            classes = sorted(glob(directory + '/*/'))
            num_classes = len(classes)
            labels = classes
            dirs = tf.data.Dataset.from_tensor_slices((classes, labels))               # 1

        elif self.params["dataset"]=="mono_en":
            # print("Monolingual:")
            directory = "/ssd_scratch/cvit/kiran/en/train_split"
            classes = sorted(glob(directory + '/*/'))
            num_classes = len(classes)
            labels = classes
            dirs = tf.data.Dataset.from_tensor_slices((classes, labels))               # 1

        elif self.params["dataset"]=="timit":
            print("Other cases:")
            directory = "/GeneralizedKWS/timit3/selected_words_train"
            classes = sorted(glob(directory + '/*/')) # final slash selects directories only
            num_classes = len(classes)
            labels = sorted(os.listdir("/GeneralizedKWS/timit3/selected_words_train"))
            dirs = tf.data.Dataset.from_tensor_slices((classes, labels))   

        elif self.params["dataset"]=="tamil":
            # print("Monolingual:")
            directory = "/ssd_scratch/cvit/kiran/tamil/train_split"
            classes = sorted(glob(directory + '/*/'))
            num_classes = len(classes)
            labels = classes
            dirs = tf.data.Dataset.from_tensor_slices((classes, labels))               # 1            # 1

        elif self.params["dataset"]=="vallander":
            # print("Monolingual:")
            directory = "/ssd_scratch/cvit/kiran/vallander/train_split"
            classes = sorted(glob(directory + '/*/'))
            num_classes = len(classes)
            labels = classes
            dirs = tf.data.Dataset.from_tensor_slices((classes, labels))               # 1            # 1

        # if self.params["dataset"] != "google":
        if self.params['shuffle']:
          dirs = dirs.shuffle(self._size)
        files = dirs.apply(tf.contrib.data.parallel_interleave(
          get_files, cycle_length=num_classes, block_length=3,      # 2
          sloppy=False))
        self._dataset = files
        
        self._dataset = self._dataset.repeat()
        self._dataset = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)
        # print(self._dataset)
        self._dataset = self._dataset.map(
            lambda line: tf.py_func(
                self._parse_audio_transcript_element,
                [line],
                [self.params['dtype'], tf.int32, tf.int32, tf.int32, tf.float32, tf.int32],
                stateful=False,
            ),
            num_parallel_calls=8,
        )
        if self.params['max_duration'] > 0:
          self._dataset = self._dataset.filter(
              lambda x, x_len, y, y_len, duration, label:
              tf.less_equal(duration, self.params['max_duration'])
          )
        if self.params['min_duration'] > 0:
          self._dataset = self._dataset.filter(
              lambda x, x_len, y, y_len, duration, label:
              tf.greater_equal(duration, self.params['min_duration'])
          )
        self._dataset = self._dataset.map(
            lambda x, x_len, y, y_len, duration, label:
            [x, x_len, y, y_len, label],
            num_parallel_calls=8,
        )
        self._dataset = self._dataset.padded_batch(
            self.params['batch_size'],
            padded_shapes=([None, self.params['num_audio_features']], 1, [None], 1, 1),
            padding_values=(
                tf.cast(0, self.params['dtype']), 0, self.target_pad_value, 0, 0),
        )
        # print(self._dataset)
      else:
        indices = self.split_data(
            np.array(list(map(str, range(len(self.all_files)))))
        )
        self._dataset = tf.data.Dataset.from_tensor_slices(
            np.hstack((indices[:, np.newaxis], self._files[:, np.newaxis]))
        )
        self._dataset = self._dataset.repeat()
        self._dataset = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)
        self._dataset = self._dataset.map(
            lambda line: tf.py_func(
                self._parse_audio_element,
                [line],
                [self.params['dtype'], tf.int32, tf.int32, tf.float32, tf.float32, tf.int32],
                stateful=False,
            ),
            num_parallel_calls=8,
        )
        if self.params['max_duration'] > 0:
          self._dataset = self._dataset.filter(
              lambda x, x_len,x2, x2_len, idx, duration, duration2:
              tf.less_equal(duration, self.params['max_duration'])
          )
        if self.params['min_duration'] > 0:
            self._dataset = self._dataset.filter(
              lambda x, x_len, x2, x2_len, y, y_len, duration, duration2:
              tf.greater_equal(duration, self.params['min_duration'])
          )
        self._dataset = self._dataset.map(
            lambda x, x_len,  x2, x2_len, idx, duration, duration2:
            [x, x_len, idx],
            num_parallel_calls=16,
        )
        self._dataset = self._dataset.padded_batch(
            self.params['batch_size'],
            padded_shapes=([None, self.params['num_audio_features']], 1, 1)
        )

      self._iterator = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)\
                           .make_initializable_iterator()

      if self.params['mode'] != 'infer':
        x, x_length, y, y_length, label = self._iterator.get_next()
        # need to explicitly set batch size dimension
        # (it is employed in the model)
        y.set_shape([self.params['batch_size'], None])
        y_length = tf.reshape(y_length, [self.params['batch_size']])
        label = tf.reshape(label, [self.params['batch_size']])
      else:
        x, x_length, x_id = self._iterator.get_next()
        x_id = tf.reshape(x_id, [self.params['batch_size']])

      x.set_shape([self.params['batch_size'], None,
                   self.params['num_audio_features']])
      x_length = tf.reshape(x_length, [self.params['batch_size']])


      pad_to = self.params.get("pad_to", 8)
      if pad_to > 0 and self.params.get('backend') == 'librosa':
        # we do padding with TF for librosa backend
        num_pad = tf.mod(pad_to - tf.mod(tf.reduce_max(x_length), pad_to), pad_to)
        x = tf.pad(x, [[0, 0], [0, num_pad], [0, 0]])

      self._input_tensors = {}
      self._input_tensors["source_tensors"] = [x, x_length]
      if self.params['mode'] != 'infer':
        self._input_tensors['target_tensors'] = [y, y_length]
        self._input_tensors['label'] = label
      else:
        self._input_tensors['source_ids'] = [x_id]
    print(self._input_tensors)


  def get_test_var(self):
    return self.testing_var

  def create_interactive_placeholders(self):
    self._x = tf.placeholder(
        dtype=self.params['dtype'],
        shape=[
            self.params['batch_size'],
            None,
            self.params['num_audio_features']
        ]
    )
    self._x_length = tf.placeholder(
        dtype=tf.int32,
        shape=[self.params['batch_size']]
    )
    self._x_id = tf.placeholder(
        dtype=tf.int32,
        shape=[self.params['batch_size']]
    )

    self._input_tensors = {}
    self._input_tensors["source_tensors"] = [self._x, self._x_length]
    self._input_tensors['source_ids'] = [self._x_id]

  def create_feed_dict(self, model_in):
    """ Creates the feed dict for interactive infer

    Args:
      model_in (str or np.array): Either a str that contains the file path of the
        wav file, or a numpy array containing 1-d wav file.

    Returns:
      feed_dict (dict): Dictionary with values for the placeholders.
    """
    audio_arr = []
    audio_length_arr = []
    x_id_arr = []
    for line in model_in:
      if isinstance(line, string_types):
        audio, audio_length, x_id, _ = self._parse_audio_element([0, line])
      elif isinstance(line, np.ndarray):
        audio, audio_length, x_id, _ = self._get_audio(line)
      else:
        raise ValueError(
            "Speech2Text's interactive inference mode only supports string or",
            "numpy array as input. Got {}". format(type(line))
        )
      audio_arr.append(audio)
      audio_length_arr.append(audio_length)
      x_id_arr.append(x_id)
    max_len = np.max(audio_length_arr)
    pad_to = self.params.get("pad_to", 8)
    if pad_to > 0 and self.params.get('backend') == 'librosa':
      max_len += (pad_to - max_len % pad_to) % pad_to
    for i, audio in enumerate(audio_arr):
      audio = np.pad(
          audio, ((0, max_len-len(audio)), (0, 0)),
          "constant", constant_values=0.
      )
      audio_arr[i] = audio

    audio = np.reshape(
        audio_arr,
        [self.params['batch_size'],
         -1,
         self.params['num_audio_features']]
    )
    audio_length = np.reshape(audio_length_arr, [self.params['batch_size']])
    x_id = np.reshape(x_id_arr, [self.params['batch_size']])

    feed_dict = {
        self._x: audio,
        self._x_length: audio_length,
        self._x_id: x_id,
    }
    return feed_dict

  def _parse_audio_transcript_element(self, element):
    """Parses tf.data element from TextLineDataset into audio and text.
    Args:
      element: tf.data element from TextLineDataset.
    Returns:
      tuple: source audio features as ``np.array``, length of source sequence,
      target text as `np.array` of ids, target text length.
    """
    # print("ELEMENT: ", element)
    # self.testing_var = element
    # if self.params["dataset"] == "google":
    #   audio_filename, transcript = element
    #   transcript = str(transcript, 'utf-8')
    # else:
    audio_filename = element
    transcript = str(element, "utf-8").split("/")[-2]
    # print(transcript)
    
    if self.params["dataset"]=="mswc":
        # print("multilingual: ")        
        language = str(element, "utf-8").split("/")[-3]
        label = self.params["labels"][language][transcript]
        processed_transcript = transcript.replace("’", "").replace("ß", "s").replace("ä", "ah").replace("ü", "u").replace("ö", "o").replace("à", "a").replace("á", "a").replace("è", "e").replace("é", "e").replace("ê", "e").replace("í", "i").replace("ñ", "n").replace("ò", "o").replace("ó", "o").replace("ù", "u")
        transcript = processed_transcript
    elif self.params["dataset"]=="nl":
        # print("Monolingual:")
        label = self.params["labels"][transcript]
        processed_transcript = transcript.replace('è', 'e').replace('é', 'e').replace('ê', 'e').replace('ë', 'e').replace('î',"i").replace('ï',"i").replace('ö', 'o').replace('û','u').replace('ü', 'u').replace('—','').replace('’',"")
        transcript = processed_transcript

    elif self.params["dataset"]=="vallander":
        # print("Monolingual:")
        label = self.params["labels"][transcript]
        processed_transcript = transcript.replace('à','a').replace('è','e').replace('ò','o').replace('ö','o').replace('ü','u').replace('’','')
        transcript = processed_transcript

    if not six.PY2:
      # transcript = str(transcript, 'utf-8')
      # print("transcript-->", transcript)
      audio_filename = str(audio_filename, 'utf-8')
      # audio_filename2 = str(audio_filename2, 'utf-8')
      # audio_filename3 = str(audio_filename3, 'utf-8')
    if self.params['bpe']:
      target_indices = self.sp.EncodeAsIds(transcript)
    else:
      target_indices = [self.params['char2idx'][c] for c in transcript]
    if self.autoregressive:
      target_indices = target_indices + [self.end_index]
    target = np.array(target_indices)

    if self.params.get("syn_enable", False):
      audio_filename = audio_filename.format(np.random.choice(self.params["syn_subdirs"]))
      # audio_filename2 = audio_filename2.format(np.random.choice(self.params["syn_subdirs"]))
      # audio_filename3 = audio_filename3.format(np.random.choice(self.params["syn_subdirs"]))

    source, audio_duration = get_speech_features_from_file(
        audio_filename,
        params=self.params
    )
    if self.params["dataset"]=="timit":
      label = self.params['labels'][transcript]
    elif self.params["dataset"] == "google":
      label = self.params["labels"].index(transcript.lower())
    elif self.params["dataset"] == "mono_en":
      label = self.params['labels'][transcript]
    elif self.params["dataset"] == "tamil":
      label = self.params['labels'][transcript]
    # elif self.params["dataset"] == "vallander":
    #   label = self.params['labels'][transcript]


    return source.astype(self.params['dtype'].as_numpy_dtype()), \
        np.int32([len(source)]), \
        np.int32(target), \
        np.int32([len(target)]), \
        np.float32([audio_duration]), \
        np.int32([label])

  def _get_audio(self, wav):
    """Parses audio from wav and returns array of audio features.
    Args:
      wav: numpy array containing wav

    Returns:
      tuple: source audio features as ``np.array``, length of source sequence,
      sample id.
    """
    source, audio_duration = get_speech_features(
        wav, 16000., self.params
    )

    return source.astype(self.params['dtype'].as_numpy_dtype()), \
        np.int32([len(source)]), np.int32([0]), \
        np.float32([audio_duration])

  def _parse_audio_element(self, id_and_audio_filename):
    """Parses audio from file and returns array of audio features.
    Args:
      id_and_audio_filename: tuple of sample id and corresponding
          audio file name.
    Returns:
      tuple: source audio features as ``np.array``, length of source sequence,
      sample id.
    """
    idx, audio_filename = id_and_audio_filename
    source, audio_duration = get_speech_features_from_file(
        audio_filename,
        params=self.params
    )
    return source.astype(self.params['dtype'].as_numpy_dtype()), \
        np.int32([len(source)]), np.int32([idx]), \
        np.float32([audio_duration])

  @property
  def input_tensors(self):
    """Dictionary with input tensors.
    ``input_tensors["source_tensors"]`` contains:
      * source_sequence
        (shape=[batch_size x sequence length x num_audio_features])
      * source_length (shape=[batch_size])
    ``input_tensors["target_tensors"]`` contains:
      * target_sequence
        (shape=[batch_size x sequence length])
      * target_length (shape=[batch_size])
    """
    return self._input_tensors

  def get_size_in_samples(self):
    """Returns the number of audio files."""
    return len(self._files)

def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Parameters
    ----------
    filename:
        File path of the data sample.
    validation_percentage:  float
        How much of the data set to use for validation. (between 0 and 1)
    testing_percentage: float
        How much of the data set to use for testing. (between 0 and 1)

    Returns
    -------
    String
        one of 'training', 'validation', or 'testing'.
    """

    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


def prepare_words_list(wanted_words):
    """Prepends common tokens to the custom word list.

    Parameters
    ----------
    wanted_words: list
     List of strings containing the custom words.

    Returns
    -------
    list
        List with the standard silence and unknown tokens added.
    """

    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words
