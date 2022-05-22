
class Config:
    img_height = 257
    num_channels = 1
    width_reduction = 8
    epochs = 100
    batch_size = 8
    channels = 1
    path_to_audios = '/home/jcms/TFG/Synthesis_procedures/piano'
    path_to_kern = '/home/jcms/TFG/GT'

    # Model
    filters = [16, 16]
    kernel_size = (3, 3)
    pool_size = [[3, 3], [3, 3]]
    pool_strides = [[1, 1], [1, 1]]