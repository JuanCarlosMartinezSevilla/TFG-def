import numpy as np
import itertools
from config import Config
import os
import joblib
import librosa
import librosa.display
#from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv2
import madmom
from madmom.audio.spectrogram import LogarithmicFilterbank, LogarithmicFilteredSpectrogram, Spectrogram


def normalize(image):
    return (255. - image) / 255.

def greedy_decoding(prediction, i2w):
    out_best = np.argmax(prediction, axis=1)
    out_best = [k for k, g in itertools.groupby(list(out_best))]
    return [i2w[s] for s in out_best if s != len(i2w)]

def levenshtein(a,b):
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

def from_spec_create_image(file_path, stft, h, w):

    if not os.path.exists(Config.path_to_temp):
        os.makedirs(Config.path_to_temp)

    SR = 22050
    n_fft = 512
    hop_length = n_fft // 4

    plt.clf()
    plt.cla()
    
    fig = plt.figure(figsize=(w/100, h/100))
    plt.axis('off')
    img = librosa.display.specshow(librosa.amplitude_to_db(stft), sr=SR, x_axis='s', y_axis='linear', hop_length=hop_length)
    path_to_save_temp_img = os.path.join(Config.path_to_temp, file_path)
    
    # Saves the image
    fig.savefig(f'{path_to_save_temp_img}.png', bbox_inches='tight', pad_inches=0)
    
    # Reads image and deletes it
    img = cv2.imread(f'{path_to_save_temp_img}.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    os.remove(f'{path_to_save_temp_img}.png')

    img = normalize(img)

    return img

memory = joblib.memory.Memory('./dataset', mmap_mode='r', verbose=1)
@memory.cache
def calculate_STFT_array_from_src (file_path: str):
    n_fft = 512
    hop_length = n_fft // 4
    eps = 0.001

    SR = 22050	# sample rate of audio
    
    audio, _ = librosa.load(file_path, sr=SR, mono=True)
    stft_complex = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    audio_shape = stft_complex.shape
    h = audio_shape[0]
    w = audio_shape[1]

    audio_options = dict(
            num_channels=1,
            sample_rate=44100,
            filterbank=LogarithmicFilterbank,
            frame_size=4096,
            fft_size=4096,
            hop_size=441 * 2,  # 25 fps -> 441 * 4 ; 50 fps -> 441 * 2
            num_bands=48,
            fmin=30,
            fmax=8000.0,
            fref=440.0,
            norm_filters=True,
            unique_filters=True,
            circular_shift=False,
            norm=True
    )

    dt = float(audio_options['hop_size']) / float(audio_options['sample_rate'])
    x = LogarithmicFilteredSpectrogram(file_path, **audio_options)
    x = np.flip(np.transpose(x),0)
    stft = (x - np.amin(x)) / (np.amax(x) - np.amin(x))

    return stft, h, w

#memory = joblib.memory.Memory('./dataset/new_stft', mmap_mode='r', verbose=1)
#@memory.cache
#def calculate_STFT_array_from_src(audiofilename):
#    audio_options = dict(
#            num_channels=1,
#            sample_rate=44100,
#            filterbank=LogarithmicFilterbank,
#            frame_size=4096,
#            fft_size=4096,
#            hop_size=441 * 2,  # 25 fps -> 441 * 4 ; 50 fps -> 441 * 2
#            num_bands=48,
#            fmin=30,
#            fmax=8000.0,
#            fref=440.0,
#            norm_filters=True,
#            unique_filters=True,
#            circular_shift=False,
#            norm=True
#    )
#
#    dt = float(audio_options['hop_size']) / float(audio_options['sample_rate'])
#    x = LogarithmicFilteredSpectrogram(audiofilename, **audio_options)
#    x = np.flip(np.transpose(x),0)
#    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
#
#    return x

memory = joblib.memory.Memory('./dataset', mmap_mode='r', verbose=0)
@memory.cache
def krn_tokenizer(f_path):

    include_key = False
    include_measure = False

    f = open(f_path, "r")
    
    tokens = []
    symbols = ['J', 'L', '[', ']', '_', ';', 'y', 'q', '\n', 'n']
    for l in f:
        if '*' in l:
            if 'k[' in l and include_key:
                tokens.append(l[:-1])
            if 'M' in l and include_measure:
                tokens.append(l[:-1])
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
    return tokens

def parse_lst(lst_path):

    X = []
    Y = []
    vocabulary = set()

    lines = open(lst_path, 'r').read().splitlines()
    for line in lines:
        line_aud = line + '.wav'
        line_kern = line + '.skm'
        #line_img = line + '.jpeg'
        audio = os.path.join(Config.path_to_audios, line_aud)
        kern = os.path.join(Config.path_to_kern, line_kern)
        #img = os.path.join(Config.path_to_img, line_img)

        #print(img)
        spectrogram = calculate_STFT_array_from_src(audio)
        tokens = krn_tokenizer(kern)
        
        #print('Forma:' ,np.amax(np.array(spectrogram)), np.amin(np.array(spectrogram)))

        for t in tokens:
            vocabulary.add(t)

        X.append(audio) # rutas de los espectrogramas
        Y.append(kern)  # rutas de los audios

    w2i = {symbol: idx for idx, symbol in enumerate(vocabulary)}
    i2w = {idx: symbol for idx, symbol in enumerate(vocabulary)}

    print(f"{len(X)} samples loaded with {len(w2i)}-sized vocabulary")
    return X, Y, w2i, i2w

