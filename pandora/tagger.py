#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import utils
import evaluation
from model import build_model
from preprocessing import Preprocessor


class Tagger():

    def __init__(self):
        self.nb_encoding_layers = 3
        self.nb_dense_dims = 300
        self.nb_epochs = 15
        self.batch_size = 100

    def fit(self, train_instances, dev_instances):

        train_tokens, train_lemmas, train_pos, train_morph = zip(*train_instances)
        dev_tokens, dev_lemmas, dev_pos, dev_morph = zip(*dev_instances)

        self.preprocessor = Preprocessor()

        train_X_focus, train_X_lemma, train_X_pos = \
            self.preprocessor.fit_transform(tokens=train_tokens,
                                               lemmas=train_lemmas,
                                               pos=train_pos,
                                               morph=train_morph)

        dev_X_focus, dev_X_lemma, dev_X_pos = \
            self.preprocessor.transform(tokens=dev_tokens,
                                        lemmas=dev_lemmas,
                                        pos=dev_pos,
                                        morph=dev_morph)
        

        self.model = build_model(token_len=self.preprocessor.max_token_len,
                                 lemma_len=self.preprocessor.max_lemma_len,
                                 token_char_vector_dict=self.preprocessor.token_char_dict,
                                 lemma_char_vector_dict=self.preprocessor.lemma_char_dict,
                                 nb_encoding_layers=self.nb_encoding_layers,
                                 nb_dense_dims=self.nb_dense_dims,
                                )

        for e in range(self.nb_epochs):
            print("-> epoch ", e+1, "...")

            # fit on train:
            d = {'focus_in': train_X_focus,
                 'lemma_out': train_X_lemma}
            self.model.fit(data=d,
                  nb_epoch = 1,
                  batch_size = self.batch_size)

            # get loss on train:
            train_loss = self.model.evaluate(data=d,
                                    batch_size=self.batch_size)
            print("\t - loss:\t{:.3}".format(train_loss))

            # get dev predictions:
            d = {'focus_in': dev_X_focus}
            predictions = self.model.predict(data=d,
                                    batch_size=self.batch_size)
            
            # convert predictions to actual strings:
            pred_lemmas = evaluation.convert_to_lemmas(\
                                predictions=predictions['lemma_out'],
                                out_char_idx=self.preprocessor.lemma_char_idx)

            
            # check a random selection
            for token, pred_lem in zip(dev_tokens[300:400], pred_lemmas[300:400]):
                if token not in self.preprocessor.known_tokens:
                    print(token, '>', pred_lem, '(UNK)')
                else:
                    print(token, '>', pred_lem)
    
    
            print('::: Dev scores :::')
            all_acc, kno_acc, unk_acc = evaluation.accuracies(gold=dev_lemmas,
                                                 silver=pred_lemmas,
                                                 test_tokens=dev_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
            print('+\tall acc:', all_acc)
            print('+\tkno acc:', kno_acc)
            print('+\tunk acc:', unk_acc)
            
            """
            print('::: Test predictions :::')
            d = {'in': test_in_X}
            predictions = m.predict(data=d, batch_size=batch_size)
    
            # convert predictions to actual strings:
            pred_lemmas = evaluation.convert_to_lemmas(predictions=predictions['out'],
                                    out_char_idx=out_char_idx)
            """
                
    
        

        
        
