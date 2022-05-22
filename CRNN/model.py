from keras.layers import Dense, LSTM, Reshape, Permute, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
import keras.backend as K
from config import Config
from keras.layers import LeakyReLU


def ctc_lambda_func(args):
    y_pred, krn_files, spec_length, krn_length = args

    return K.ctc_batch_cost(krn_files, y_pred, spec_length, krn_length)

def build_model(vocabulary_size):
    # input with shape of height and width 
    inputs = Input(shape=(Config.img_height, None, Config.num_channels), dtype="float32", name="Audio")
    x = inputs

    conv_filters = Config.filters
    pool_k_size = Config.pool_size
    conv_k_size = Config.kernel_size
    pool_strides = Config.pool_strides
    number_poolings = len(Config.filters)
    last_number_filters = conv_filters[len(Config.filters)-1]
    lstm_units = [256, 256]

    for idx, f in enumerate(conv_filters):
        x = Conv2D(f, conv_k_size, activation = 'LeakyReLU', padding='same',
        name = f'Conv{idx+1}' )(x)    
        x = BatchNormalization(name=f'BatchNorm{idx+1}')(x)
        x = MaxPool2D(pool_size=pool_k_size[idx], strides=pool_strides[idx], padding='same',
        name = f'MaxPool{idx+1}')(x)

    ## Preparing CNN output to use it on Recurrent layers
    x = Permute((2, 1, 3))(x) # (b, h, w, c) => (b, w, h, c)
    target_shape = (-1, Config.img_height//(2**number_poolings) * last_number_filters) # (b, w, h, c) ==> (b, w, h*c)
    x = Reshape(target_shape=target_shape, name='reshape')(x)

    # bidirectional LSTM layers with units=128
    for units in lstm_units:
        x = Bidirectional(LSTM(units, return_sequences=True, dropout = 0.2))(x)

    outputs = Dense(vocabulary_size+1, activation = 'softmax')(x)

    model_val = Model(inputs=inputs, outputs=outputs)

    krn_files = Input(name='krn_files', shape=[None], dtype='int64')
    spec_length = Input(name='input_length', shape=[1], dtype='int64')
    krn_length = Input(name='krn_length', shape=[1], dtype='int64')
    
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, krn_files, spec_length, krn_length])

                        # spec    krn        len(spec)    len(krn)
    model = Model(inputs=[inputs, krn_files, spec_length, krn_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')
    model.summary()

    return model, model_val