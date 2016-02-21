#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

from keras.utils import np_utils

import utils
import codecs
import evaluation
from model import build_model
from preprocessing import Preprocessor
from pretraining import Pretrainer


class Tagger():

    def __init__(self):
        self.nb_encoding_layers = 3
        self.nb_dense_dims = 300
        self.batch_size = 128
        self.nb_left_tokens = 2
        self.nb_right_tokens = 1
        self.nb_context_tokens = self.nb_left_tokens + self.nb_right_tokens
        self.nb_embedding_dims = 150
        self.include_token = True
        self.include_context = True
        self.include_lemma = True
        self.include_pos = True
        self.include_morph = True

        self.setup = False
        self.nb_epochs = 0

    def setup_for_fit(self, train_instances, dev_instances, unseen_tokens):

        self.train_tokens, self.train_lemmas, self.train_pos, self.train_morph = \
            zip(*train_instances)
        self.dev_tokens, self.dev_lemmas, self.dev_pos, self.dev_morph = \
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

        # all contextual stuff:
        self.pretrainer = Pretrainer(nb_left_tokens=self.nb_left_tokens,
                                     nb_right_tokens=self.nb_right_tokens,
                                     size=self.nb_embedding_dims)
        self.pretrained_embeddings, self.train_token_vocab = \
            self.pretrainer.fit(tokens=self.train_tokens)

        self.train_contexts = self.pretrainer.transform(tokens=self.train_tokens)
        self.dev_contexts = self.pretrainer.transform(tokens=self.dev_tokens)
        
        self.unseen_tokens = None
        if unseen_tokens:
            self.unseen_tokens = unseen_tokens
            self.unseen_X_focus = self.preprocessor.transform(tokens=self.unseen_tokens)
            self.unseen_contexts = self.pretrainer.transform(tokens=self.unseen_tokens)

        print('Building model...')
        self.model = build_model(token_len=self.preprocessor.max_token_len,
                                 token_char_vector_dict=self.preprocessor.token_char_dict,
                                 lemma_len=self.preprocessor.max_lemma_len,
                                 nb_tags=len(self.preprocessor.pos_encoder.classes_),
                                 nb_morph_cats=len(self.preprocessor.morph_encoder.classes_),
                                 lemma_char_vector_dict=self.preprocessor.lemma_char_dict,
                                 nb_encoding_layers=self.nb_encoding_layers,
                                 nb_dense_dims=self.nb_dense_dims,
                                 nb_embedding_dims=self.nb_embedding_dims,
                                 nb_train_tokens=len(self.train_token_vocab),
                                 nb_context_tokens=self.nb_context_tokens,
                                 pretrained_embeddings=self.pretrained_embeddings,
                                 include_token=self.include_token,
                                 include_context=self.include_context,
                                 include_lemma=self.include_lemma,
                                 include_pos=self.include_pos,
                                 include_morph=self.include_morph,
                                )

        self.setup = True


    def epoch(self):
        if not self.setup:
            raise ValueError('Not set up yet... Call Tagger.setup_for_fit() first.')

        # update nb of epochs ran so far:
        self.nb_epochs += 1
        print("-> epoch ", self.nb_epochs, "...")

        # update learning rate at specific points:
        if self.nb_epochs % 10 == 0:
            self.model.optimizer.lr.set_value(np.float32(self.model.optimizer.lr.get_value() * 0.33))
        
        # fit on train:
        d = {}
        if self.include_token:
            d['focus_in'] = self.train_X_focus
        if self.include_context:
            d['context_in'] = self.train_contexts
        if self.include_lemma:
            d['lemma_out'] = self.train_X_lemma
        if self.include_pos:
            d['pos_out'] = self.train_X_pos
        if self.include_morph:
            d['morph_out'] = self.train_X_morph
        
        self.model.fit(data=d,
              nb_epoch = 1,
              batch_size = self.batch_size)

        # get loss on train:
        train_loss = self.model.evaluate(data=d,
                                batch_size=self.batch_size)
        print("\t - total train loss:\t{:.3}".format(train_loss))

        # get dev predictions:
        d = {}
        if self.include_token:
            d['focus_in'] = self.dev_X_focus
        if self.include_context:
            d['context_in'] = self.dev_contexts

        preds = self.model.predict(data=d,
                                batch_size=self.batch_size)
        
        # convert predictions to actual strings:
        if self.include_lemma:
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(predictions=preds['lemma_out'])
        if self.include_pos:
            pred_pos = self.preprocessor.inverse_transform_pos(predictions=preds['pos_out'])
        if self.include_morph:
            pred_morph = self.preprocessor.inverse_transform_morph(predictions=preds['morph_out'])

        """
        # check a random selection
        for token, pred_lem, pred_p, pred_m in zip(self.dev_tokens[300:400],
                                                   pred_lemmas[300:400],
                                                   pred_pos[300:400],
                                                   pred_morph[300:400]):
            if token not in self.preprocessor.known_tokens:
                print(token, '>', pred_lem, pred_p, pred_m, '(UNK)')
            else:
                print(token, '>', pred_lem, pred_p, pred_m)
        """

        if self.include_lemma:
            print('::: Dev scores (lemmas) :::')
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.dev_lemmas,
                                                 silver=pred_lemmas,
                                                 test_tokens=self.dev_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)

        if self.include_pos:
            print('::: Dev scores (pos) :::')
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.dev_pos,
                                                 silver=pred_pos,
                                                 test_tokens=self.dev_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
        
        if self.include_morph:     
            print('::: Dev scores (morph) :::')
            #all_acc, kno_acc, unk_acc = evaluation.multilabel_accuracies(gold=self.dev_morph,
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.dev_morph,
                                                 silver=pred_morph,
                                                 test_tokens=self.dev_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
        """
        ##############################################################################
        if self.unseen_tokens:
            # unseen, completely new data data:
            d = {'focus_in': self.unseen_X_focus}
            predictions = self.model.predict(data=d,
                                    batch_size=self.batch_size)
            
            # convert predictions to actual strings:
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(\
                                predictions=predictions['lemma_out'])
            pred_pos = self.preprocessor.inverse_transform_pos(\
                                predictions=predictions['pos_out'])
            pred_morph = self.preprocessor.inverse_transform_morph(\
                                predictions=predictions['morph_out'])
            
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
        """




        


                
    
        

        
        
