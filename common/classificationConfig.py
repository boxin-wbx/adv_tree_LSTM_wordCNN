import os
from common.Global import Global

class classificationConfig:

    dev_dir = Global.external_tools + '/dev/'
    train_dir = Global.external_tools + '/train/'
    test_dir = Global.external_tools + '/test/'
    token_file_labels = [train_dir, dev_dir, test_dir]
    # qa vocab
    vocab = Global.external_tools + '/imdb.vocab'
    embed = Global.external_tools + '/imdb_embed.pth'

    # glove
    glove = Global.data_files + '/glove/glove.840B.300d'
