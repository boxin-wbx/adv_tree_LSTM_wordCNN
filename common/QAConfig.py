from common.Global import Global


class QAConfig:
    train_data = Global.external_tools + '/squad/data_train.json'
    train_share = Global.external_tools + '/squad/shared_train.json'
    # share
    test_data = Global.external_tools + '/squad/test_train.json'
    test_share = Global.external_tools + '/squad/test_train.json'

    # tree train
    tree_train = Global.external_tools + '/qa/tree/train.json'
    tree_path = Global.external_tools + '/qa/tree/'

    # qa vocab
    vocab = Global.external_tools + '/qa/qa.vocab'
    embed = Global.external_tools + '/qa/embed.pth'

    # glove
    glove = Global.external_tools + '/glove/glove.6B.300d'

