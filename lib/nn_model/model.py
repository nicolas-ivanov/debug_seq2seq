import os.path

from keras.models import Sequential
from seq2seq.models import SimpleSeq2seq

from configs.config import TOKEN_REPRESENTATION_SIZE, HIDDEN_LAYER_DIMENSION, SAMPLES_BATCH_SIZE, \
    INPUT_SEQUENCE_LENGTH, ANSWER_MAX_TOKEN_LENGTH, NN_MODEL_PATH
from utils.utils import get_logger

_logger = get_logger(__name__)


def get_nn_model(token_dict_size):
    _logger.info('Initializing NN model with the following params:')
    _logger.info('Input dimension: %s (token vector size)' % TOKEN_REPRESENTATION_SIZE)
    _logger.info('Hidden dimension: %s' % HIDDEN_LAYER_DIMENSION)
    _logger.info('Output dimension: %s (token dict size)' % token_dict_size)
    _logger.info('Input seq length: %s ' % INPUT_SEQUENCE_LENGTH)
    _logger.info('Output seq length: %s ' % ANSWER_MAX_TOKEN_LENGTH)
    _logger.info('Batch size: %s' % SAMPLES_BATCH_SIZE)

    model = Sequential()
    seq2seq = SimpleSeq2seq(
        input_dim=TOKEN_REPRESENTATION_SIZE,
        input_length=INPUT_SEQUENCE_LENGTH,
        hidden_dim=HIDDEN_LAYER_DIMENSION,
        output_dim=token_dict_size,
        output_length=ANSWER_MAX_TOKEN_LENGTH,
        depth=1
    )

    model.add(seq2seq)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # use previously saved model if it exists
    _logger.info('Looking for a model %s' % NN_MODEL_PATH)

    if os.path.isfile(NN_MODEL_PATH):
        _logger.info('Loading previously calculated weights...')
        model.load_weights(NN_MODEL_PATH)

    _logger.info('Model is built')
    return model
