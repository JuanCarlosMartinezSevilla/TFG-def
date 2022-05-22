
class Config:
    img_height = 257
    num_channels = 1
    width_reduction = 4
    epochs = 100
    batch_size = 8
    channels = 1
    path_to_audios = '/workspace/CRNN/TFG/Synthesis_procedures/piano'
    path_to_kern = '/workspace/CRNN/TFG/GT'

    # Model
    filters = [16, 16]
    kernel_size = (3, 3)
    pool_size = [[3, 3], [3, 3]]
    pool_strides = [[1, 1], [1, 1]]
