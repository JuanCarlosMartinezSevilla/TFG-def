
class Config:
    img_height = 229
    num_channels = 1
    width_reduction = 4
    epochs = 200
    batch_size = 2
    path_to_audios = '/workspace/CRNN/TFG/Synthesis_procedures/piano'
    path_to_kern = '/workspace/CRNN/TFG/GT'
    path_to_temp = '/workspace/CRNN/temp'

    # LOCAL training
    #path_to_audios = '../piano'
    #path_to_kern = '../GT'

    # Model
    #filters = [16, 16]
    #kernel_size = [3, 3]
    #pool_size = [3, 3]
    #pool_strides = [1, 1]


    filters = [16, 16]
    kernel_size = [[3,3], [3,3]]
    pool_size = [[3,3], [3,3]]
    pool_strides = [[1, 1], [1, 1]]
    activations = ['LeakyReLU', 'LeakyReLU']
    param_activation = [0.2, 0.2]
    batch_norm = [True, True]
    #Recurrent stages=
    units = [256, 256]
    batch_norm_rec = [True, True]
    dropout = [0, 0]
