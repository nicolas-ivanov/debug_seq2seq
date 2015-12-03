import numpy as np
from configs.config import TOKEN_REPRESENTATION_SIZE


def get_token_vector(token, model):
    if token in model.vocab:
        return np.array(model[token])

    # return a zero vector for the words that are not presented in the model
    return np.zeros(TOKEN_REPRESENTATION_SIZE)


def get_vectorized_token_sequence(sequence, w2v_model, max_sequence_length, reverse=False):
    vectorized_token_sequence = np.zeros((max_sequence_length, TOKEN_REPRESENTATION_SIZE), dtype=np.float)

    for i, word in enumerate(sequence):
        vectorized_token_sequence[i] = get_token_vector(word, w2v_model)

    if reverse:
        # reverse token vectors order in input sequence as suggested here
        # http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
        vectorized_token_sequence = vectorized_token_sequence[::-1]

    return vectorized_token_sequence
