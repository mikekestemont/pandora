#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import cPickle as pickle
import os
import codecs
import shutil
from operator import itemgetter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

from keras.utils import np_utils
from keras.models import model_from_json

import editdistance

import utils
import evaluation
from model import build_model
from preprocessing import Preprocessor
from pretraining import Pretrainer


class Tagger():

    def __init__(self, nb_encoding_layers = 1,
                    nb_dense_dims = 30,
                    batch_size = 100,
                    nb_left_tokens = 2,
                    nb_right_tokens = 2,
                    nb_embedding_dims = 150,
                    model_name = 'new_model',
                    postcorrect = True,
                    include_token = True,
                    include_context = True,
                    include_lemma = True,
                    include_pos = True,
                    include_morph = True,
                    include_dev = True,
                    include_test = True,
                    nb_filters = 100,
                    filter_length = 3,
                    focus_repr = 'recurrent',
                    dropout_level = .1,
                    ):

        # set hyperparameters:
        self.nb_encoding_layers = nb_encoding_layers
        self.nb_dense_dims = nb_dense_dims
        self.batch_size = batch_size
        self.nb_left_tokens = nb_left_tokens
        self.nb_right_tokens = nb_right_tokens
        self.nb_context_tokens = self.nb_left_tokens + self.nb_right_tokens
        self.nb_embedding_dims = nb_embedding_dims
        self.model_name = model_name
        self.postcorrect = postcorrect
        self.nb_filters = nb_filters
        self.filter_length = filter_length
        self.focus_repr = focus_repr
        self.dropout_level = dropout_level

        # which subnets?
        self.include_token = include_token
        self.include_context = include_context

        # which headnets?
        self.include_lemma = include_lemma
        self.include_pos = include_pos
        self.include_morph = include_morph

        # include dev and/or test?
        self.include_dev = include_dev
        self.include_test = include_test
        
        # initialize:
        self.setup = False
        self.nb_epochs = 0

        # initialize paths:
        # create a models directory:
        MODELS_DIR = 'models'
        if not os.path.isdir(MODELS_DIR):
            os.mkdir(MODELS_DIR)
        self.model_path = os.sep.join((MODELS_DIR, self.model_name))

        self.train_tokens, self.dev_tokens, self.test_tokens = None, None, None
        self.train_lemmas, self.dev_lemmas, self.test_lemmas = None, None, None
        self.train_pos, self.dev_pos, self.test_pos = None, None, None
        self.train_morph, self.dev_morph, self.test_morph = None, None, None

    def load(self):
        print('Re-loading preprocessor...')
        self.preprocessor = pickle.load(open(os.sep.join((self.model_path, \
                                    'preprocessor.p')), 'rb'))
        print('Re-loading pretrainer...')
        self.pretrainer = pickle.load(open(os.sep.join((self.model_path, \
                                    'pretrainer.p')), 'rb'))
        print('Re-building model...')
        self.model = model_from_json(open(os.sep.join((self.model_path, 'model_architecture.json'))).read())
        self.model.load_weights(os.sep.join((self.model_path, 'model_weights.hdf5')))
        print('Loading known lemmas...')
        self.known_lemmas = pickle.load(open(os.sep.join((self.model_path, \
                                    'known_lemmas.p')), 'rb'))

    def setup_to_train(self, train_data=None, dev_data=None, test_data=None):
        # create a model directory:
        if os.path.isdir(self.model_path):
            shutil.rmtree(self.model_path)
        os.mkdir(self.model_path)

        self.train_tokens = train_data['token']
        if self.include_test:
            self.test_tokens = test_data['token']
        if self.include_dev:
            self.dev_tokens = dev_data['token']
        if self.include_lemma:
            self.train_lemmas = train_data['lemma']
            self.known_lemmas = set(self.train_lemmas)
            if self.include_dev:
                self.dev_lemmas = dev_data['lemma']            
            if self.include_test:
                self.test_lemmas = test_data['lemma']
        if self.include_pos:
            self.train_pos = train_data['pos']
            if self.include_dev:
                self.dev_pos = dev_data['pos']
            if self.include_test:
                self.test_pos = test_data['pos']
        if self.include_morph:
            self.train_morph = train_data['morph']
            if self.include_dev:
                self.dev_morph = dev_data['morph']
            if self.include_test:
                self.test_morph = test_data['morph']

        self.preprocessor = Preprocessor().fit(tokens=self.train_tokens,
                                               lemmas=self.train_lemmas,
                                               pos=self.train_pos,
                                               morph=self.train_morph,
                                               include_lemma=self.include_lemma,
                                               include_morph=self.include_morph)
        self.pretrainer = Pretrainer(nb_left_tokens=self.nb_left_tokens,
                                     nb_right_tokens=self.nb_right_tokens,
                                     size=self.nb_embedding_dims)
        self.pretrainer.fit(tokens=self.train_tokens)

        train_transformed = self.preprocessor.transform(tokens=self.train_tokens,
                                               lemmas=self.train_lemmas,
                                               pos=self.train_pos,
                                               morph=self.train_morph)
        if self.include_dev:
            dev_transformed = self.preprocessor.transform(tokens=self.dev_tokens,
                                        lemmas=self.dev_lemmas,
                                        pos=self.dev_pos,
                                        morph=self.dev_morph)
        if self.include_test:
            test_transformed = self.preprocessor.transform(tokens=self.test_tokens,
                                        lemmas=self.test_lemmas,
                                        pos=self.test_pos,
                                        morph=self.test_morph)

        self.train_X_focus = train_transformed['X_focus']
        if self.include_dev:
            self.dev_X_focus = dev_transformed['X_focus']
        if self.include_test:
            self.test_X_focus = test_transformed['X_focus']

        if self.include_lemma:
            self.train_X_lemma = train_transformed['X_lemma']
            if self.include_dev:
                self.dev_X_lemma = dev_transformed['X_lemma']
            if self.include_test:
                self.test_X_lemma = test_transformed['X_lemma']

        if self.include_pos:
            self.train_X_pos = train_transformed['X_pos']
            if self.include_dev:
                self.dev_X_pos = dev_transformed['X_pos']
            if self.include_test:
                self.test_X_pos = test_transformed['X_pos']

        if self.include_morph:
            self.train_X_morph = train_transformed['X_morph']
            if self.include_dev:
                self.dev_X_morph = dev_transformed['X_morph']
            if self.include_test:
                self.test_X_morph = test_transformed['X_morph']

        self.train_contexts = self.pretrainer.transform(tokens=self.train_tokens)
        if self.include_dev:
            self.dev_contexts = self.pretrainer.transform(tokens=self.dev_tokens)
        if self.include_test:
            self.test_contexts = self.pretrainer.transform(tokens=self.test_tokens)
        
        print('Building model...')
        nb_tags = None
        try:
            nb_tags = len(self.preprocessor.pos_encoder.classes_)
        except AttributeError:
            pass
        nb_morph_cats = None
        try:
            nb_morph_cats = self.preprocessor.nb_morph_cats
        except AttributeError:
            pass
        max_token_len, token_char_dict = None, None
        try:
            max_token_len = self.preprocessor.max_token_len
            token_char_dict = self.preprocessor.token_char_dict
        except AttributeError:
            pass
        max_lemma_len, lemma_char_dict = None, None
        try:
            max_lemma_len = self.preprocessor.max_lemma_len
            lemma_char_dict = self.preprocessor.lemma_char_dict
        except AttributeError:
            pass
        nb_lemmas = None
        try:
            nb_lemmas = len(self.preprocessor.lemma_encoder.classes_)
        except AttributeError:
            pass
        self.model = build_model(token_len=max_token_len,
                             token_char_vector_dict=token_char_dict,
                             lemma_len=max_lemma_len,
                             nb_tags=nb_tags,
                             nb_morph_cats=nb_morph_cats,
                             lemma_char_vector_dict=lemma_char_dict,
                             nb_encoding_layers=self.nb_encoding_layers,
                             nb_dense_dims=self.nb_dense_dims,
                             nb_embedding_dims=self.nb_embedding_dims,
                             nb_train_tokens=len(self.pretrainer.train_token_vocab),
                             nb_context_tokens=self.nb_context_tokens,
                             pretrained_embeddings=self.pretrainer.pretrained_embeddings,
                             include_token=self.include_token,
                             include_context=self.include_context,
                             include_lemma=self.include_lemma,
                             include_pos=self.include_pos,
                             include_morph=self.include_morph,
                             nb_filters = self.nb_filters,
                             filter_length = self.filter_length,
                             focus_repr = self.focus_repr,
                             dropout_level = self.dropout_level,
                             nb_lemmas = nb_lemmas,
                            )
        self.save()
        self.setup = True

    def print_stats(self):
        print('Train stats:')
        utils.stats(tokens=self.train_tokens, lemmas=self.train_lemmas, known=self.preprocessor.known_tokens)
        print('Test stats:')
        utils.stats(tokens=self.test_tokens, lemmas=self.test_lemmas, known=self.preprocessor.known_tokens)

    def test(self, multilabel_threshold=0.5):
        if not self.include_test:
            raise ValueError('Please do not call .test() if no test data is available.')

        score_dict = {}

        # get test predictions:
        d = {}
        if self.include_token:
            d['focus_in'] = self.test_X_focus
        if self.include_context:
            d['context_in'] = self.test_contexts

        test_preds = self.model.predict(data=d,
                                batch_size=self.batch_size)

        if self.include_lemma:
            print('::: Test scores (lemmas) :::')
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(predictions=test_preds['lemma_out'])
            if self.postcorrect:
                for i in range(len(pred_lemmas)):
                    if pred_lemmas[i] not in self.known_lemmas:
                        pred_lemmas[i] = min(self.known_lemmas,
                                        key=lambda x: editdistance.eval(x, pred_lemmas[i]))
            score_dict['test_lemma'] = evaluation.single_label_accuracies(gold=self.test_lemmas,
                                                 silver=pred_lemmas,
                                                 test_tokens=self.test_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)

        if self.include_pos:
            print('::: Test scores (pos) :::')
            pred_pos = self.preprocessor.inverse_transform_pos(predictions=test_preds['pos_out'])
            score_dict['test_pos'] = evaluation.single_label_accuracies(gold=self.test_pos,
                                                 silver=pred_pos,
                                                 test_tokens=self.test_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
        
        if self.include_morph:     
            print('::: Test scores (morph) :::')
            pred_morph = self.preprocessor.inverse_transform_morph(predictions=test_preds['morph_out'],
                                                                   threshold=multilabel_threshold)
            if self.include_morph == 'label':
                score_dict['test_morph'] = evaluation.single_label_accuracies(gold=self.test_morph,
                                                 silver=pred_morph,
                                                 test_tokens=self.test_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)                
            elif self.include_morph == 'multilabel':
                score_dict['test_morph'] = evaluation.multilabel_accuracies(gold=self.test_morph,
                                                 silver=pred_morph,
                                                 test_tokens=self.test_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
        return score_dict

    def save(self):
        # save architecture:
        json_string = self.model.to_json()
        with open(os.sep.join((self.model_path, 'model_architecture.json')), 'wb') as f:
            f.write(json_string)
        # save weights:
        self.model.save_weights(os.sep.join((self.model_path, 'model_weights.hdf5')), \
                overwrite=True)
        # save preprocessor:
        with open(os.sep.join((self.model_path, 'preprocessor.p')), 'wb') as f:
            pickle.dump(self.preprocessor, f)
        # save pretrainer:
        with open(os.sep.join((self.model_path, 'pretrainer.p')), 'wb') as f:
            pickle.dump(self.pretrainer, f)
        # save known lemmas:
        with open(os.sep.join((self.model_path, 'known_lemmas.p')), 'wb') as f:
            pickle.dump(self.known_lemmas, f)
        
        # plot current embeddings:
        if self.include_context:
            weights = self.model.nodes['context_embedding'].get_weights()[0]
            X = np.array([weights[self.pretrainer.train_token_vocab.index(w), :] \
                    for w in self.pretrainer.mfi], dtype='float32')
            # dimension reduction:
            tsne = TSNE(n_components=2)
            coor = tsne.fit_transform(X) # unsparsify
            plt.clf(); sns.set_style('dark')
            sns.plt.rcParams['axes.linewidth'] = 0.4
            fig, ax1 = sns.plt.subplots()  
            labels = self.pretrainer.mfi
            # first plot slices:
            x1, x2 = coor[:,0], coor[:,1]
            ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
            # clustering on top (add some colouring):
            clustering = AgglomerativeClustering(linkage='ward',
                            affinity='euclidean', n_clusters=8)
            clustering.fit(coor)
            # add names:
            for x, y, name, cluster_label in zip(x1, x2, labels, clustering.labels_):
                ax1.text(x, y, name, ha='center', va="center",
                         color=plt.cm.spectral(cluster_label / 10.),
                         fontdict={'family': 'Arial', 'size': 8})
            # control aesthetics:
            ax1.set_xlabel(''); ax1.set_ylabel('')
            ax1.set_xticklabels([]); ax1.set_xticks([])
            ax1.set_yticklabels([]); ax1.set_yticks([])
            sns.plt.savefig(os.sep.join((self.model_path, 'embed_after.pdf')),
                            bbox_inches=0)

    def epoch(self, autosave=True):
        if not self.setup:
            raise ValueError('Not set up yet... Call Tagger.setup_() first.')

        # update nb of epochs ran so far:
        self.nb_epochs += 1
        print("-> epoch ", self.nb_epochs, "...")

        # update learning rate at specific points:
        if self.nb_epochs % 10 == 0:
            old_lr  = self.model.optimizer.lr.get_value()
            new_lr = np.float32(old_lr * 0.5)
            self.model.optimizer.lr.set_value(new_lr)
            print('\t- Lowering learning rate > was:', old_lr, ', now:', new_lr)

        # fit on train:
        full_train_d = {}
        if self.include_token:
            full_train_d['focus_in'] = self.train_X_focus
        if self.include_context:
            full_train_d['context_in'] = self.train_contexts
        if self.include_lemma:
            full_train_d['lemma_out'] = self.train_X_lemma
        if self.include_pos:
            full_train_d['pos_out'] = self.train_X_pos
        if self.include_morph:
            full_train_d['morph_out'] = self.train_X_morph
        
        self.model.fit(data=full_train_d,
              nb_epoch = 1,
              shuffle = True,
              batch_size = self.batch_size)

        # get train loss:
        train_loss = self.model.evaluate(data=full_train_d,
                                batch_size=self.batch_size)
        print("\t - total train loss:\t{:.3}".format(train_loss))

        # get train preds:
        train_preds = self.model.predict(data=full_train_d,
                                batch_size=self.batch_size)

        if self.include_dev:
            # get dev predictions:
            d = {}
            if self.include_token:
                d['focus_in'] = self.dev_X_focus
            if self.include_context:
                d['context_in'] = self.dev_contexts

            dev_preds = self.model.predict(data=d,
                                    batch_size=self.batch_size)

        score_dict = {}
        if self.include_lemma:
            print('::: Train scores (lemmas) :::')
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(predictions=train_preds['lemma_out'])
            score_dict['train_lemma'] = evaluation.single_label_accuracies(gold=self.train_lemmas,
                                                 silver=pred_lemmas,
                                                 test_tokens=self.train_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
            if self.include_dev:
                print('::: Dev scores (lemmas) :::')
                pred_lemmas = self.preprocessor.inverse_transform_lemmas(predictions=dev_preds['lemma_out'])
                if self.postcorrect:
                    for i in range(len(pred_lemmas)):
                        if pred_lemmas[i] not in self.known_lemmas:
                            pred_lemmas[i] = min(self.known_lemmas,
                                            key=lambda x: editdistance.eval(x, pred_lemmas[i]))
                score_dict['dev_lemma'] = evaluation.single_label_accuracies(gold=self.dev_lemmas,
                                                     silver=pred_lemmas,
                                                     test_tokens=self.dev_tokens,
                                                     known_tokens=self.preprocessor.known_tokens)

        if self.include_pos:
            print('::: Train scores (pos) :::')
            pred_pos = self.preprocessor.inverse_transform_pos(predictions=train_preds['pos_out'])
            score_dict['train_pos'] = evaluation.single_label_accuracies(gold=self.train_pos,
                                                 silver=pred_pos,
                                                 test_tokens=self.train_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
            if self.include_dev:
                print('::: Dev scores (pos) :::')
                pred_pos = self.preprocessor.inverse_transform_pos(predictions=dev_preds['pos_out'])
                score_dict['dev_pos'] = evaluation.single_label_accuracies(gold=self.dev_pos,
                                                     silver=pred_pos,
                                                     test_tokens=self.dev_tokens,
                                                     known_tokens=self.preprocessor.known_tokens)
        
        if self.include_morph:
            print('::: Train scores (morph) :::')
            pred_morph = self.preprocessor.inverse_transform_morph(predictions=train_preds['morph_out'])
            if self.include_morph == 'label':
                score_dict['train_morph'] = evaluation.single_label_accuracies(gold=self.train_morph,
                                                 silver=pred_morph,
                                                 test_tokens=self.train_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
            elif self.include_morph == 'multilabel':
                score_dict['train_morph'] = evaluation.multilabel_accuracies(gold=self.train_morph,
                                                 silver=pred_morph,
                                                 test_tokens=self.train_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)


            if self.include_dev:
                print('::: Dev scores (morph) :::')
                pred_morph = self.preprocessor.inverse_transform_morph(predictions=dev_preds['morph_out'])
                if self.include_morph == 'label':
                    score_dict['dev_morph'] = evaluation.single_label_accuracies(gold=self.train_morph,
                                                     silver=pred_morph,
                                                     test_tokens=self.dev_tokens,
                                                     known_tokens=self.preprocessor.known_tokens)
                elif self.include_morph == 'multilabel':
                    score_dict['dev_morph'] = evaluation.multilabel_accuracies(gold=self.train_morph,
                                                     silver=pred_morph,
                                                     test_tokens=self.dev_tokens,
                                                     known_tokens=self.preprocessor.known_tokens)

        if autosave:
            self.save()

        return score_dict



    def annotate(self, tokens):
        X_focus = self.preprocessor.transform(tokens=tokens)['X_focus']
        X_context = self.pretrainer.transform(tokens=tokens)

        # get predictions:
        d = {}
        if self.include_token:
            d['focus_in'] = X_focus
        if self.include_context:
            d['context_in'] = X_context
        preds = self.model.predict(data=d, batch_size=self.batch_size)

        annotation_dict = {'tokens': tokens}
        if self.include_lemma:
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(predictions=preds['lemma_out'])
            annotation_dict['lemmas'] = pred_lemmas
            if self.postcorrect:
                for i in range(len(pred_lemmas)):
                    if pred_lemmas[i] not in self.known_lemmas:
                        pred_lemmas[i] = min(self.known_lemmas,
                                            key=lambda x: editdistance.eval(x, pred_lemmas[i]))
                annotation_dict['postcorrect_lemmas'] = pred_lemmas

        if self.include_pos:
            pred_pos = self.preprocessor.inverse_transform_pos(predictions=preds['pos_out'])
            annotation_dict['pos'] = pred_pos
        
        if self.include_morph:
            pred_morph = self.preprocessor.inverse_transform_morph(predictions=preds['morph_out'])
            annotation_dict['morph'] = pred_morph

        return annotation_dict
        

        


                
    
        

        
        
