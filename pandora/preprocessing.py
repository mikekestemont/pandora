#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from operator import itemgetter

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from keras.utils import np_utils
import numpy as np

from model import build_model

def index_characters(tokens, v2u=True):
    if v2u:
        vocab = {ch for tok in tokens for ch in tok.lower().replace('v', 'u')}
    else:
        vocab = {ch for tok in tokens for ch in tok.lower()}
    vocab = vocab.union({'$', '|'})
    char_vocab = tuple(sorted(vocab))
    char_vector_dict, char_idx = {}, {}
    filler = np.zeros(len(char_vocab), dtype='float32')

    for idx, char in enumerate(char_vocab):
        ph = filler.copy()
        ph[idx] = 1
        char_vector_dict[char] = ph
        char_idx[idx] = char

    return char_vector_dict, char_idx

def vectorize_tokens(tokens, char_vector_dict,
                     max_len=15, v2u=True):
    X = []
    for token in tokens:
        token = token.lower()
        if v2u:
            token.lower().replace('v', 'u')
        x = vectorize_token(seq=token,
                            char_vector_dict=char_vector_dict,
                            max_len=max_len)
        X.append(x)

    return np.asarray(X, dtype='float32')


def vectorize_lemmas(lemmas, char_vector_dict,
                     max_len=15):
    X = []
    for lemma in lemmas:
        lemma = lemma.lower()
        x = vectorize_lemma(seq=lemma,
                            char_vector_dict=char_vector_dict,
                            max_len=max_len)
        X.append(x)

    X = np.asarray(X, dtype='float32')
    return X

def vectorize_token(seq, char_vector_dict, max_len):
    # cut, if needed:
    seq = seq[:(max_len - 1)]
    seq += '|'

    while len(seq) < max_len:
        seq = seq + '$'

    seq = seq[::-1] # reverse order (cf. paper)!

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
    seq = seq[:(max_len - 1)]
    seq += '|'
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
            for a in ml.split('|'):
                k, v = a.split('=')
                d[k] = v
        except ValueError:
            pass
        morph_dicts.append(d)
    return morph_dicts


class Preprocessor():

    def __init__(self):
        pass

    def fit(self, tokens, lemmas, pos, morph, include_lemma, include_morph):
        # fit focus tokens:
        self.max_token_len = len(max(tokens, key=len))+1
        self.token_char_dict, self.token_char_idx = \
            index_characters(tokens)
        self.known_tokens = set(tokens)
        
        # fit lemmas:
        if lemmas:
            self.include_lemma = include_lemma
            self.known_lemmas = set(lemmas)
            if include_lemma == 'generate':
                self.max_lemma_len = len(max(lemmas, key=len))+1
                self.lemma_char_dict, self.lemma_char_idx = \
                    index_characters(lemmas)
            elif include_lemma == 'label':
                self.lemma_encoder = LabelEncoder()
                self.lemma_encoder.fit(lemmas + ['<UNK>']) # do we need this?

        # fit pos labels:
        if pos:
            self.pos_encoder = LabelEncoder()
            self.pos_encoder.fit(pos + ['<UNK>']) # do we need this?

        if morph:
            self.include_morph = include_morph
            if self.include_morph == 'label':
                self.morph_encoder = LabelEncoder()
                self.morph_encoder.fit(morph + ['<UNK>'])
                self.nb_morph_cats = len(self.morph_encoder.classes_)
            elif self.include_morph == 'multilabel':
                # fit morph analysis:
                morph_dicts = parse_morphs(morph)
                self.morph_encoder = DictVectorizer(sparse=False)
                self.morph_encoder.fit(morph_dicts)
                self.nb_morph_cats = len(self.morph_encoder.feature_names_)
                self.morph_idxs = {}
                for i, feat_name in enumerate(self.morph_encoder.feature_names_):
                    label, _ = feat_name.strip().split('=')
                    try:
                        self.morph_idxs[label].add(i)
                    except KeyError:
                        self.morph_idxs[label] = set()
                        self.morph_idxs[label].add(i)
        
        return self

    def transform(self, tokens=None, lemmas=None,
                  pos=None, morph=None):
        # vectorize focus tokens:
        X_focus = vectorize_tokens(\
                    tokens=tokens,
                    char_vector_dict=self.token_char_dict,
                    max_len=self.max_token_len)
        returnables = {'X_focus': X_focus}

        if lemmas and self.include_lemma:
            if self.include_lemma == 'generate':
                # vectorize lemmas:
                X_lemma = vectorize_lemmas(\
                            lemmas=lemmas,
                            char_vector_dict=self.lemma_char_dict,
                            max_len=self.max_lemma_len)

            elif self.include_lemma == 'label':
                lemmas = [l if l in self.lemma_encoder.classes_ \
                        else '<UNK>' for l in lemmas]
                lemmas = self.lemma_encoder.transform(lemmas)
            
                X_lemma = np_utils.to_categorical(lemmas,
                        nb_classes=len(self.lemma_encoder.classes_))

            returnables['X_lemma'] = X_lemma

        if pos:
            # vectorize pos:
            pos = [p if p in self.pos_encoder.classes_ \
                        else '<UNK>' for p in pos]
            pos = self.pos_encoder.transform(pos)
            
            X_pos = np_utils.to_categorical(pos,
                        nb_classes=len(self.pos_encoder.classes_))
            returnables['X_pos'] = X_pos

        if morph:
            if self.include_morph == 'label':
                morph = [m if m in self.morph_encoder.classes_ \
                        else '<UNK>' for m in morph]
                morph = self.morph_encoder.transform(morph)
            
                X_morph = np_utils.to_categorical(morph,
                        nb_classes=len(self.morph_encoder.classes_))
                returnables['X_morph'] = X_morph

            elif self.include_morph == 'multilabel':
                # vectorize morph:
                morph_dicts = parse_morphs(morph)
                X_morph = self.morph_encoder.transform(morph_dicts)
                returnables['X_morph'] = X_morph

        return returnables

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

        if self.include_lemma == 'generate':
            for pred in predictions:
                pred_lem = ''
                for positions in pred:
                    top_idx = np.argmax(positions) # winning position
                    c = self.lemma_char_idx[top_idx] # look up corresponding char
                    if c in ('|', '$'):
                        break
                    else:
                        pred_lem += c # add character
                pred_lemmas.append(pred_lem)

        elif self.include_lemma == 'label':
            predictions = np.argmax(predictions, axis=1)
            pred_lemmas = self.lemma_encoder.inverse_transform(predictions)    
        
        return pred_lemmas

    def inverse_transform_pos(self, predictions):
        """
        """
        predictions = np.argmax(predictions, axis=1)
        return self.pos_encoder.inverse_transform(predictions)

    def inverse_transform_morph(self, predictions, threshold=.5):
        """
        Only select highest activation per category, if that max
        is above threshold.
        """
        if self.include_morph == 'label':
            predictions = np.argmax(predictions, axis=1)
            return self.morph_encoder.inverse_transform(predictions)
        elif self.include_morph == 'multilabel':
            morphs = []
            for pred in predictions:
                m = []
                for label, idxs in self.morph_idxs.items():
                    scores = ((pred[idx], idx) for idx in idxs)
                    max_score = max(scores, key=itemgetter(0))
                    if max_score[0] >= threshold:
                        m.append(self.morph_encoder.feature_names_[max_score[1]])
                if m:
                    morphs.append('|'.join(sorted(set(m))))
                else:
                    morphs.append('_')
        return morphs
        
