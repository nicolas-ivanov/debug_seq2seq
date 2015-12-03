import sys
import os
from itertools import tee

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dialog_processor import get_processed_dialog_lines_and_index_to_token
from configs.config import CORPUS_PATH, PROCESSED_CORPUS_PATH, TOKEN_INDEX_PATH, W2V_PARAMS
from lib.w2v_model import w2v
from lib.nn_model.model import get_nn_model
from lib.nn_model.predict import predict_sentence
from utils.utils import get_logger

_logger = get_logger(__name__)


def predict():
    # preprocess the dialog and get index for its vocabulary
    processed_dialog_lines, index_to_token = \
        get_processed_dialog_lines_and_index_to_token(CORPUS_PATH, PROCESSED_CORPUS_PATH, TOKEN_INDEX_PATH)

    # dualize iterator
    dialog_lines_for_w2v, dialog_lines_for_nn = tee(processed_dialog_lines)
    _logger.info('-----')

    # use gensim realisatino of word2vec instead of keras embeddings due to extra flexibility
    w2v_model = w2v.get_dialogs_model(W2V_PARAMS, dialog_lines_for_w2v)
    _logger.info('-----')

    nn_model = get_nn_model(token_dict_size=len(index_to_token))

    while True:
        input_sentence = raw_input('> ')
        predict_sentence(input_sentence, nn_model, w2v_model, index_to_token)


if __name__ == '__main__':
    predict()
