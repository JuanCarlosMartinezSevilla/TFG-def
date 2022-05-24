import numpy as np
import itertools
from config import Config
import os
import joblib
import librosa
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt


def normalize(image):
    return (255. - image) / 255.

def greedy_decoding(prediction, i2w):
    out_best = np.argmax(prediction, axis=1)
    out_best = [k for k, g in itertools.groupby(list(out_best))]
    return [i2w[s] for s in out_best if s != len(i2w)]


def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


memory = joblib.memory.Memory('./dataset', mmap_mode='r', verbose=0)
@memory.cache
def calculate_STFT_array_from_src (file_path: str) -> np.array:
    n_fft = 512
    hop_length = n_fft // 4
    eps = 0.001

    SR = 22050	# sample rate of audio
    
    audio, _ = librosa.load(file_path, sr=SR, mono=True)
    stft_complex = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    log_stft = np.log(np.abs(stft_complex) + eps)
    log_stft = np.flipud(log_stft)

    print(log_stft, log_stft.shape)

    return log_stft

memory = joblib.memory.Memory('./dataset', mmap_mode='r', verbose=0)
@memory.cache
def krn_tokenizer(f_path):

    f = open(f_path, "r")
    
    tokens = []
    symbols = ['J', 'L', '[', ']', '_', ';', 'y', 'q', '\n', 'n']
    for l in f:
        if '*' in l:
            continue   
        if '!' in l:
            continue  
        if 'rr' in l:
            tokens.append('rr')
            continue
        if '=' in l:
            tokens.append('=')
            continue

        for s in symbols:
            if s in l:
                l = l.replace(s, '')
        tokens.append(l)
    print(tokens)
    return tokens

def parse_lst(lst_path):

    X = []
    Y = []
    vocabulary = set()

    lines = open(lst_path, 'r').read().splitlines()
    for line in tqdm(lines):
        line_aud = line + '.wav'
        line_kern = line + '.skm'
        audio = os.path.join(Config.path_to_audios, line_aud)
        kern = os.path.join(Config.path_to_kern, line_kern)

        spectrogram = calculate_STFT_array_from_src(audio)
        tokens = krn_tokenizer(kern)

        #print('Forma:' ,spectrogram.shape)

        for t in tokens:
            vocabulary.add(t)

        X.append(audio) # rutas de los espectrogramas
        Y.append(kern)  # rutas de los audios

    w2i = {symbol: idx for idx, symbol in enumerate(vocabulary)}
    i2w = {idx: symbol for idx, symbol in enumerate(vocabulary)}

    print("{} samples loaded with {}-sized vocabulary".format(len(X), len(w2i)))
    return X, Y, w2i, i2w

