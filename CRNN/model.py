import tensorflow as tf
from config import Config


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_model(vocabulary_size):
    input = tf.keras.layers.Input(shape=(Config.img_height, None, Config.num_channels))

    x = input

    ### Convolutional block ###
    for conv_index in range(len(Config.filters)):
        x = tf.keras.layers.Conv2D(
            filters = Config.filters[conv_index],
            kernel_size = Config.kernel_size[conv_index],
            padding = 'same',
            name = 'Conv' + str(conv_index + 1)
        )(x)
        if Config.batch_norm[conv_index]:
                x = tf.keras.layers.BatchNormalization(
                    name = 'BatchNorm' + str(conv_index + 1)
                )(x)
        
        x = tf.keras.layers.MaxPool2D(
            pool_size = Config.pool_size[conv_index],
            strides = Config.pool_strides[conv_index],
            padding = 'same',
            name = 'MaxPool' + str(conv_index + 1)
        )(x)

    ### Reshaping block ###
    x = tf.keras.layers.Permute((2, 1, 3))(x)
    x_shape = x.shape
    x = tf.keras.layers.Reshape(target_shape=(-1, (x_shape[3]*x_shape[2])), name='Reshape')(x)

    ### Recurrent block ###
    for rec_index in range(len(Config.units)):
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            Config.units[rec_index], 
            return_sequences=True, 
            dropout=0.25))(x)
        x = tf.keras.layers.BatchNormalization(
            name = 'BatchNormRec' + str(rec_index + 1))(x)


    y_pred= tf.keras.layers.Dense(vocabulary_size + 1, activation='softmax', name='Dense')(x) 

    model_pr = tf.keras.Model(inputs=input, outputs=y_pred)
    #model_pr.summary()

    labels = tf.keras.layers.Input(name='the_labels', shape=[None], dtype='float32')
    input_length = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')

    loss_out = tf.keras.layers.Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([y_pred, labels, input_length, label_length])

    model_tr = tf.keras.Model(inputs=[input, labels, input_length, label_length],
                              outputs=loss_out)

    model_tr.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
    #model_tr.summary()

    return model_tr, model_pr


if __name__ == "__main__":
    pass
