#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from operator import itemgetter

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
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

def parse_morphs(morph):
    morph_dicts = []
    for ml in morph:
        d = {}
        try:
            for a in ml:
                k, v = a.split('=')
                d[k] = v
        except ValueError:
            pass
        morph_dicts.append(d)
    return morph_dicts


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

        # fit idx for 'known' tokens and lemmas:
        self.known_tokens = set(tokens)
        self.known_lemmas = set(lemmas)

        # fit morph analysis:
        morph_dicts = parse_morphs(morph)
        self.morph_encoder = DictVectorizer(sparse=False)
        self.morph_encoder.fit(morph_dicts)

        self.morph_idxs = {}
        for i, feat_name in enumerate(self.morph_encoder.feature_names_):
            label, _ = feat_name.strip().split('=')
            try:
                self.morph_idxs[label].add(i)
            except KeyError:
                self.morph_idxs[label] = set()
                self.morph_idxs[label].add(i)

        return self

    def transform(self, tokens, lemmas=None,
                  pos=None, morph=None):
        # vectorize focus tokens:
        X_focus = vectorize_tokens(\
                    tokens=tokens,
                    char_vector_dict=self.token_char_dict,
                    max_len=self.max_token_len)
        returnables = [X_focus]

        if lemmas:
            # vectorize lemmas:
            X_lemma = vectorize_lemmas(\
                        lemmas=lemmas,
                        char_vector_dict=self.lemma_char_dict,
                        max_len=self.max_lemma_len)
            returnables.append(X_lemma)

        if pos:
            # vectorize pos:
            pos = [p if p in self.pos_encoder.classes_\
                        else '<UNK>' for p in pos] # do we need this?
            pos = self.pos_encoder.transform(pos)
            
            X_pos = np_utils.to_categorical(pos,
                        nb_classes=len(self.pos_encoder.classes_))
            returnables.append(X_pos)

        if morph:
            # vectorize morph:
            morph_dicts = parse_morphs(morph)
            X_morph = self.morph_encoder.transform(morph_dicts)
            returnables.append(X_morph)

        if len(returnables) > 1:
            return returnables
        else:
            return returnables[0]

    def fit_transform(self, tokens, lemmas, pos, morph):
        self.fit(tokens, lemmas, pos, morph)
        return self.transform(tokens, lemmas, pos, morph)


    def inverse_transform_lemmas(self, predictions):
        """
        For each prediction, convert 2D char matrix
        with probabilities to actual lemmas, using
        character index for the output strings.
        """
        pred_lemmas = []

        for pred in predictions:
            pred_lem = ''
            for positions in pred:
                top_idx = np.argmax(positions) # winning position
                c = self.lemma_char_idx[top_idx] # look up corresponding char
                if c == '$':
                    break
                else:
                    pred_lem += c # add character
            pred_lemmas.append(pred_lem)

        return pred_lemmas

    def inverse_transform_pos(self, predictions):
        """
        """
        predictions = np.argmax(predictions, axis=1)
        return self.pos_encoder.inverse_transform(predictions)

    def inverse_transform_morph(self, predictions, threshold=1.0):
        """
        Only select highest activation per category, if that max
        is above threshold.
        """
        morphs = []
        for pred in predictions:
            m = []
            for label, idxs in self.morph_idxs.items():
                scores = ((pred[idx], idx) for idx in idxs)
                max_score = max(scores, key=itemgetter(0))
                if max_score[0] >= threshold:
                    m.append(self.morph_encoder.feature_names_[max_score[1]])
            print(m)
            if m:
                morphs.append(m)
            else:
                morphs.append(['_'])
        return morphs
        
