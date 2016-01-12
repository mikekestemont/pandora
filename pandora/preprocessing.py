#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np

from model import build_model

def index_characters(tokens):
    tokens = tokens + ('$',) # add end of token symbol
    char_vocab = tuple({ch for tok in tokens for ch in tok})
    char_vector_dict, char_idx = {}, {}
    filler = np.zeros(len(char_vocab), dtype='float32')

    for idx, char in enumerate(char_vocab):
        ph = filler.copy()
        ph[idx] = 1
        char_vector_dict[char] = ph
        char_idx[idx] = char

    return char_vector_dict, char_idx

def vectorize_tokens(tokens, char_vector_dict,
                     max_len=15):
    X = []
    for token in tokens:
        x = vectorize_token(seq=token,
                            char_vector_dict=char_vector_dict,
                            max_len=max_len)
        X.append(x)

    return np.asarray(X, dtype='float32')


def vectorize_lemmas(lemmas, char_vector_dict,
                     max_len=15):
    X = []
    for lemma in lemmas:
        x = vectorize_lemma(seq=lemma,
                            char_vector_dict=char_vector_dict,
                            max_len=max_len)
        X.append(x)

    X = np.asarray(X, dtype='float32')
    return X

def vectorize_token(seq, char_vector_dict, max_len):
    # cut, if needed:
    seq = seq[:max_len]
    seq = seq[::-1] # reverse order (cf. paper)!

    # pad, if needed:
    while len(seq) < max_len:
        seq = '$'+seq

    seq_X = []
    filler = np.zeros(len(char_vector_dict), dtype='float32')
    for char in seq:
        try:
            seq_X.append(char_vector_dict[char])
        except KeyError:
            seq_X.append(filler)

    return np.vstack(seq_X)

def vectorize_lemma(seq, char_vector_dict, max_len):
    # cut, if needed:
    seq = seq[:max_len]
    # pad, if needed:
    while len(seq) < max_len:
        seq += '$'

    seq_X = []
    filler = np.zeros(len(char_vector_dict), dtype='float32')

    for char in seq:
        try:
            seq_X.append(char_vector_dict[char])
        except KeyError:
            seq_X.append(filler)

    return np.vstack(seq_X)


class Preprocessor():

    def __init__(self):
        pass

    def fit(self, tokens, lemmas, pos, morph):
        # fit focus tokens:
        self.max_token_len = len(max(lemmas, key=len))
        self.token_char_dict, self.token_char_idx = \
            index_characters(tokens)

        # fit lemmas:
        self.max_lemma_len = len(max(tokens, key=len))
        self.lemma_char_dict, self.lemma_char_idx = \
            index_characters(lemmas)

        # fit pos labels:
        self.pos_encoder = LabelEncoder()
        self.pos_encoder.fit(pos+('<UNK>',)) # do we need this?

        self.known_tokens = set(tokens)
        self.known_lemmas = set(lemmas)

        return self

    def transform(self, tokens, lemmas, pos, morph):
        # vectorize focus tokens:
        X_focus = vectorize_tokens(\
                    tokens=tokens,
                    char_vector_dict=self.token_char_dict,
                    max_len=self.max_token_len)

        # vectorize lemmas:
        X_lemma = vectorize_lemmas(\
                    lemmas=lemmas,
                    char_vector_dict=self.lemma_char_dict,
                    max_len=self.max_lemma_len)

        # vectorize pos:
        pos = [p if p in self.pos_encoder.classes_\
                    else '<UNK>' for p in pos] # do we need this?
        pos = self.pos_encoder.transform(pos)
        
        X_pos = np_utils.to_categorical(pos,
                    nb_classes=len(self.pos_encoder.classes_))

        return X_focus, X_lemma, X_pos

    def fit_transform(self, tokens, lemmas, pos, morph):
        self.fit(tokens, lemmas, pos, morph)
        return self.transform(tokens, lemmas, pos, morph)
