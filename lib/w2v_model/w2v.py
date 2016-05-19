import os
from itertools import tee

from gensim.models import Word2Vec

from utils.utils import get_logger

_logger = get_logger(__name__)


def _train_model(tokenized_lines, params):
    params_str = '_w' + str(params['win_size']) + '_m' + str(params['min_w_num']) + '_v' + str(params['vect_size'])

    _logger.info('Word2Vec model will be trained now. It can take long, so relax and have fun')
    _logger.info('Parameters for training: %s' % params_str)

    tokenized_lines_for_voc, tokenized_lines_for_train = tee(tokenized_lines)

    model = Word2Vec(window=int(params['win_size']),
                     min_count=int(params['min_w_num']),
                     max_vocab_size=int(params['vocab_max_size']),
                     size=int(params['vect_size']),
                     workers=int(params['workers_num']))

    model.build_vocab(tokenized_lines_for_voc)
    model.train(tokenized_lines_for_train)

    return model


def _save_model(model, model_filename):
    _logger.info('Trained model will now be saved as %s for later use' % model_filename)
    model.save(model_filename)


def load_model(full_bin_name):
    _logger.info('Loading model from %s' % full_bin_name)
    model = Word2Vec.load(full_bin_name)
    _logger.info('Model "%s" has been loaded.' % os.path.basename(full_bin_name))
    return model


def get_dialogs_model(params, tokenized_lines):
    params_str = '_w' + str(params['win_size']) + '_m' + str(params['min_w_num']) + '_v' + str(params['vect_size'])
    model_name = params['corpus_name'] + params_str + '.bin'
    full_bin_name = os.path.join(params['save_path'], params['new_models_dir'], model_name)

    if not os.path.isfile(full_bin_name):
        # bin model is not present on the disk, so get it
        model = _train_model(tokenized_lines, params)
        _save_model(model, full_bin_name)
    else:
        # bin model is on the disk, load it
        model = load_model(full_bin_name)

    return model
