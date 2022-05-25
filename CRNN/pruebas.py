import os
from utils import krn_tokenizer, calculate_STFT_array_from_src
import time
from config import Config

def token_test():
    
    path = '../GT'

    for l in os.listdir(path):
        print(l)
        print(krn_tokenizer(os.path.join(path, l))) 
        time.sleep(1)

def parse_lst(lst_path):

    X = []
    Y = []
    vocabulary = set()

    suma_longitudes = 0

    lines = open(lst_path, 'r').read().splitlines()
    for line in lines:
        line_aud = line + '.wav'
        line_kern = line + '.skm'
        audio = os.path.join(Config.path_to_audios, line_aud)
        kern = os.path.join(Config.path_to_kern, line_kern)

        spectrogram = calculate_STFT_array_from_src(audio)
        print(f"Forma: {spectrogram.shape}")
        suma_longitudes += spectrogram.shape[1]
        tokens = krn_tokenizer(kern)


        for t in tokens:
            vocabulary.add(t)

        X.append(audio) # rutas de los espectrogramas
        Y.append(kern)  # rutas de los audios

    w2i = {symbol: idx for idx, symbol in enumerate(vocabulary)}
    i2w = {idx: symbol for idx, symbol in enumerate(vocabulary)}

    print(suma_longitudes / len(lines))

    print(f"{len(X)} samples loaded with {len(w2i)}-sized vocabulary")
    return X, Y, w2i, i2w


if __name__ == '__main__':

    _, _, _, _ = parse_lst('5-crossval/train_gt_fold0.dat')
