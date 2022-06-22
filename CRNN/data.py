import numpy as np
from config import Config
import utils as U


class DataGenerator:

    def __init__(self, dataset_list_path, batch_size, num_channels, width_reduction):
        self.X, self.Y, self.w2i, self.i2w = U.parse_lst(dataset_list_path)
        self.batch_size = batch_size
        self.width_reduction = width_reduction
        self.num_channels = num_channels
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        X_batch = []
        Y_batch = []

        max_image_width = 0
        for _ in range(self.batch_size):
          

            stft, h, w = U.calculate_STFT_array_from_src(self.X[self.idx])
            sample_image = U.from_spec_create_image(self.X[self.idx], stft, h, w)
            #sample_image = U.normalize(sample_image)
            max_image_width = max(max_image_width, sample_image.shape[1])
            
            sample_image = np.expand_dims(np.array(sample_image), -1)
            X_batch.append(sample_image)
            Y_batch.append([self.w2i[symbol] for symbol in U.krn_tokenizer(self.Y[self.idx])])
            self.idx = (self.idx + 1) % len(self.X)

        X_train = np.zeros(
            shape=[self.batch_size, Config.img_height, max_image_width, self.num_channels],
            dtype=np.float32)
        L_train = np.zeros(shape=[len(X_batch), 1])

        for i, sample in enumerate(X_batch):
            X_train[i, 0:sample.shape[0], 0:sample.shape[1]] = sample
            #print(sample.shape[1])
            L_train[i] = sample.shape[1] // self.width_reduction  # width_reduction from CRNN
            #print(L_train[i])

        # Y_train, T_train
        max_length_seq = max([len(w) for w in Y_batch])

        Y_train = np.zeros(shape=[len(X_batch), max_length_seq])
        T_train = np.zeros(shape=[len(X_batch), 1])
        for i, seq in enumerate(Y_batch):
            Y_train[i, 0:len(seq)] = seq
            T_train[i] = len(seq)

        print()
        for index in range(Config.batch_size):

            print(X_train[index].shape, Y_train[index].shape, L_train[index].shape, T_train[index].shape)
        print("===========================================")
        return [X_train, Y_train, L_train, T_train], np.zeros((X_train.shape[0], 1), dtype='float16')


if __name__ == "__main__":
    pass
