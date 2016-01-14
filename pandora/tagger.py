#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from keras.utils import np_utils

import utils
import codecs
import evaluation
from model import build_model
from preprocessing import Preprocessor


class Tagger():

    def __init__(self):
        self.nb_encoding_layers = 3
        self.nb_dense_dims = 150
        self.nb_epochs = 0
        self.batch_size = 100
        self.setup = False

    def setup_for_fit(self, train_instances, dev_instances, unseen_tokens, include_morph=True):
        if include_morph:
            self.train_tokens, self.train_lemmas, self.train_pos, self.train_morph = \
                zip(*train_instances)
            self.dev_tokens, self.dev_lemmas, self.dev_pos, self.dev_morph = \
                zip(*dev_instances)
        else:
            self.train_tokens, self.train_lemmas, self.train_pos = \
                zip(*train_instances)
            self.dev_tokens, self.dev_lemmas, self.dev_pos = \
                zip(*dev_instances)

        self.preprocessor = Preprocessor()

        self.train_X_focus, self.train_X_lemma, self.train_X_pos, self.train_X_morph = \
            self.preprocessor.fit_transform(tokens=self.train_tokens,
                                               lemmas=self.train_lemmas,
                                               pos=self.train_pos,
                                               morph=self.train_morph)

        self.dev_X_focus, self.dev_X_lemma, self.dev_X_pos, self.dev_X_morph = \
            self.preprocessor.transform(tokens=self.dev_tokens,
                                        lemmas=self.dev_lemmas,
                                        pos=self.dev_pos,
                                        morph=self.dev_morph)
        try:
            nb_morph_cats = train_X_morph.shape[1]
        except:
            nb_morph_cats = None

        self.unseen_tokens = unseen_tokens
        self.unseen_X_focus = self.preprocessor.transform(tokens=self.unseen_tokens)

        print('Building model...')
        self.model = build_model(token_len=self.preprocessor.max_token_len,
                                 lemma_len=self.preprocessor.max_lemma_len,
                                 nb_tags=len(self.preprocessor.pos_encoder.classes_),
                                 nb_morph_cats=nb_morph_cats,
                                 token_char_vector_dict=self.preprocessor.token_char_dict,
                                 lemma_char_vector_dict=self.preprocessor.lemma_char_dict,
                                 nb_encoding_layers=self.nb_encoding_layers,
                                 nb_dense_dims=self.nb_dense_dims,
                                )
        self.setup = True

    def epoch(self):
        if not self.setup:
            raise ValueError('Not set up yet...')

        print("-> epoch ", self.nb_epochs+1, "...")

        # fit on train:
        d = {'focus_in': self.train_X_focus,
             'lemma_out': self.train_X_lemma,
             'pos_out': self.train_X_pos,
             #'morph_out': train_X_morph,
             }
        self.model.fit(data=d,
              nb_epoch = 1,
              batch_size = self.batch_size)
        # get loss on train:
        train_loss = self.model.evaluate(data=d,
                                batch_size=self.batch_size)
        print("\t - loss:\t{:.3}".format(train_loss))

        # get dev predictions:
        d = {'focus_in': self.dev_X_focus}
        predictions = self.model.predict(data=d,
                                batch_size=self.batch_size)
        
        # convert predictions to actual strings:
        pred_lemmas = self.preprocessor.inverse_transform_lemmas(\
                            predictions=predictions['lemma_out'])
        pred_pos = self.preprocessor.inverse_transform_pos(\
                            predictions=predictions['pos_out'])
        #pred_morph = self.preprocessor.inverse_transform_morph(\
        #                    predictions=predictions['morph_out'])
        
        # check a random selection
        for token, pred_lem, pred_p in zip(self.dev_tokens[300:400], pred_lemmas[300:400], pred_pos[300:400]):
            if token not in self.preprocessor.known_tokens:
                print(token, '>', pred_p, pred_lem, '(UNK)')
            else:
                print(token, '>', pred_p, pred_lem)


        print('::: Dev scores (lemmas) :::')
        all_acc, kno_acc, unk_acc = evaluation.accuracies(gold=self.dev_lemmas,
                                             silver=pred_lemmas,
                                             test_tokens=self.dev_tokens,
                                             known_tokens=self.preprocessor.known_tokens)
        print('+\tall acc:', all_acc)
        print('+\tkno acc:', kno_acc)
        print('+\tunk acc:', unk_acc)
        print('::: Dev scores (pos) :::')
        all_acc, kno_acc, unk_acc = evaluation.accuracies(gold=self.dev_pos,
                                             silver=pred_pos,
                                             test_tokens=self.dev_tokens,
                                             known_tokens=self.preprocessor.known_tokens)
        print('+\tall acc:', all_acc)
        print('+\tkno acc:', kno_acc)
        print('+\tunk acc:', unk_acc)

        # unseen, completely new data data:
        d = {'focus_in': self.unseen_X_focus}
        predictions = self.model.predict(data=d,
                                batch_size=self.batch_size)
        
        # convert predictions to actual strings:
        pred_lemmas = self.preprocessor.inverse_transform_lemmas(\
                            predictions=predictions['lemma_out'])
        pred_pos = self.preprocessor.inverse_transform_pos(\
                            predictions=predictions['pos_out'])
        #pred_morph = self.preprocessor.inverse_transform_morph(\
        #                    predictions=predictions['morph_out'])
        
        # print a random selection
        for token, pred_lem, pred_p in zip(self.unseen_tokens[300:400], pred_lemmas[300:400], pred_pos[300:400]):
            if token not in self.preprocessor.known_tokens:
                print(token, '>', pred_p, pred_lem, '(UNK)')
            else:
                print(token, '>', pred_p, pred_lem)

        # save results:
        with codecs.open('meld_output.txt', 'w', 'utf8') as f:
            for token, pred_lem, pred_p in zip(self.unseen_tokens, pred_lemmas, pred_pos):
                if token not in self.preprocessor.known_tokens:
                    f.write('\t'.join((token, pred_p, pred_lem, '(UNK)'))+'\n')
                else:
                    f.write('\t'.join((token, pred_p, pred_lem))+'\n')

        # update nb of epochs ran so far:
        self.nb_epochs += 1



        


                
    
        

        
        
