from tabnanny import verbose
import utils as U
from config import Config
import numpy as np

class ModelEvaluator:

    def __init__(self, set):
        self.X, self.Y = set

    def eval(self, model, i2w):
        acc_ed = 0
        acc_len = 0
        acc_count = 0

        for idx in range(len(self.X)):

            stft, h, w = U.calculate_STFT_array_from_src(self.X[self.idx])
            sample_image = U.from_spec_create_image(self.X[self.idx], stft, h, w)
            #sample_image = U.normalize(sample_image)
            sample_image = np.expand_dims(np.array(sample_image), -1)

            batch_sample = np.zeros(
                shape=[1,Config.img_height, sample_image.shape[1], Config.num_channels],
                dtype=np.float32)

            batch_sample[0] = sample_image
            prediction = model.predict(batch_sample, verbose=0)[0]

            h = U.greedy_decoding(prediction, i2w)

            acc_ed += U.levenshtein(h, self.Y[idx])
            acc_len += len(self.Y[idx])
            acc_count += 1

        return 100.0*acc_ed/acc_len
