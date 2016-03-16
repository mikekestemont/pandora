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

MODELS_DIR = 'models'
if not os.path.isdir(MODELS_DIR):
    os.mkdir(MODELS_DIR)


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
                    complex_pos = False):

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

        # which subnets?
        self.include_token = include_token
        self.include_context = include_context

        # which headnets?
        self.include_lemma = include_lemma
        self.include_pos = include_pos
        self.include_morph = include_morph

        self.complex_pos = complex_pos
        
        # initialize:
        self.setup = False
        self.nb_epochs = 0

    def setup_(self, train_data, test_data, load_pickles=False):
        # create a model directory:
        self.model_path = os.sep.join((MODELS_DIR, self.model_name))
        if os.path.isdir(self.model_path):
            shutil.rmtree(self.model_path)
        os.mkdir(self.model_path)

        self.train_tokens, self.test_tokens = None, None
        self.train_lemmas, self.test_lemmas = None, None
        self.train_pos, self.test_pos = None, None
        self.train_morph, self.test_morph = None, None

        self.train_tokens = train_data['token']
        self.test_tokens = test_data['token']
        if self.include_lemma:
            self.train_lemmas = train_data['lemma']
            self.test_lemmas = test_data['lemma']
            self.known_lemmas = set(self.train_lemmas)
        if self.include_pos:
            self.train_pos = train_data['pos']
            self.test_pos = test_data['pos']
        if self.include_morph:
            self.train_morph = train_data['morph']
            self.test_morph = test_data['morph']

        if self.complex_pos:
            self.train_pos = tuple('-'.join(i) for i in zip(self.train_pos, self.train_morph))
            #self.dev_pos = tuple('-'.join(i) for i in zip(self.dev_pos, self.dev_morph))
            self.test_pos = tuple('-'.join(i) for i in zip(self.test_pos, self.test_morph))

        print(len(self.train_tokens), '+++')
        print(len(self.train_lemmas), '!!!')
        print(len(self.train_pos), '!!!')

        if not load_pickles:
            self.preprocessor = Preprocessor().fit(tokens=self.train_tokens,
                                                   lemmas=self.train_lemmas,
                                                   pos=self.train_pos,
                                                   morph=self.train_morph)
            self.pretrainer = Pretrainer(nb_left_tokens=self.nb_left_tokens,
                                         nb_right_tokens=self.nb_right_tokens,
                                         size=self.nb_embedding_dims)
            self.pretrainer.fit(tokens=self.train_tokens)
                
        else:
            self.preprocessor = pickle.load(open(os.sep.join((self.model_path, \
                                    'preprocessor.p')), 'rb'))
            self.pretrainer = pickle.load(open(os.sep.join((self.model_path, \
                                    'pretrainer.p')), 'rb'))

        print(len(self.train_tokens), '!!!')
        print(len(self.train_lemmas), '!!!')
        print(len(self.train_pos), '!!!')
        print(len(self.train_morph), 'yyyy')

        train_transformed = self.preprocessor.transform(tokens=self.train_tokens,
                                               lemmas=self.train_lemmas,
                                               pos=self.train_pos,
                                               morph=self.train_morph)
        test_transformed = self.preprocessor.transform(tokens=self.test_tokens,
                                        lemmas=self.test_lemmas,
                                        pos=self.test_pos,
                                        morph=self.test_morph)

        self.train_X_focus = train_transformed['X_focus']
        self.test_X_focus = test_transformed['X_focus']
        if self.include_lemma:
            self.train_X_lemma = train_transformed['X_lemma']
            self.test_X_lemma = test_transformed['X_lemma']
        if self.include_pos:
            self.train_X_pos = train_transformed['X_pos']
            self.test_X_pos = test_transformed['X_pos']
        if self.include_morph:
            self.train_X_morph = train_transformed['X_morph']
            self.test_X_morph = test_transformed['X_morph']

        self.train_contexts = self.pretrainer.transform(tokens=self.train_tokens)
        self.test_contexts = self.pretrainer.transform(tokens=self.test_tokens)

        #self.print_stats()
        
        if load_pickles:
            print('Re-building model...')
            self.model = model_from_json(open(os.sep.join((self.model_path, 'model_architecture.json'))).read())
            self.model.load_weights(os.sep.join((self.model_path, 'model_weights.hdf5')))
        else:
            print('Building model...')
            nb_tags = None
            try:
                nb_tags = len(self.preprocessor.pos_encoder.classes_)
            except AttributeError:
                pass
            nb_morph_cats = None
            try:
                nb_morph_cats = len(self.preprocessor.morph_encoder.classes_)
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
                                )
        self.save()
        self.setup = True

    def print_stats(self):
        print('Train stats:')
        utils.stats(tokens=self.train_tokens, lemmas=self.train_lemmas, known=self.preprocessor.known_tokens)
        print('Test stats:')
        utils.stats(tokens=self.test_tokens, lemmas=self.test_lemmas, known=self.preprocessor.known_tokens)

    def test(self):
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
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.test_lemmas,
                                                 silver=pred_lemmas,
                                                 test_tokens=self.test_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)

        if self.include_pos:
            print('::: Test scores (pos) :::')
            pred_pos = self.preprocessor.inverse_transform_pos(predictions=test_preds['pos_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.test_pos,
                                                 silver=pred_pos,
                                                 test_tokens=self.test_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
        
        if self.include_morph:     
            print('::: Test scores (morph) :::')
            pred_morph = self.preprocessor.inverse_transform_morph(predictions=test_preds['morph_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.test_morph,
                                                 silver=pred_morph,
                                                 test_tokens=self.test_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)

        ##### OUT #######################################################################################
        if self.include_lemma and not self.include_pos:
            with codecs.open(os.sep.join((self.model_path, 'test_out.txt')), 'w', 'utf8') as f:
                for p in zip(self.test_tokens, self.test_lemmas, pred_lemmas):
                    if p[0] not in self.preprocessor.known_tokens:
                        p = list(p)
                        p.append('<UNK>')
                    try:
                        f.write('\t'.join([str(r) for r in p]) + '\n')
                    except:
                        pass
        elif not self.include_lemma and self.include_pos:
            with codecs.open(os.sep.join((self.model_path, 'test_out.txt')), 'w', 'utf8') as f:
                for p in zip(self.test_tokens, self.test_pos, pred_pos):
                    if p[0] not in self.preprocessor.known_tokens:
                        p = list(p)
                        p.append('<UNK>')
                    try:
                        f.write('\t'.join([str(r) for r in p]) + '\n')
                    except:
                        pass
        elif self.include_lemma and self.include_pos:
            with codecs.open(os.sep.join((self.model_path, 'test_out.txt')), 'w', 'utf8') as f:
                for p in zip(self.test_tokens, self.test_pos, pred_pos, self.test_lemmas, pred_lemmas):
                    if p[0] not in self.preprocessor.known_tokens:
                        p = list(p)
                        p.append('<UNK>')
                    try:
                        f.write('\t'.join([str(r) for r in p]) + '\n')
                    except:
                        pass
        return

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

    def epoch(self):
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
        
        print(self.train_X_focus.shape)
        print(self.train_contexts.shape)
        print(self.train_X_lemma.shape)
        print(self.train_X_pos.shape)

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

        """
        # get dev predictions:
        d = {}
        if self.include_token:
            d['focus_in'] = self.dev_X_focus
        if self.include_context:
            d['context_in'] = self.dev_contexts

        dev_preds = self.model.predict(data=d,
                                batch_size=self.batch_size)
        """

        if self.include_lemma:
            print('::: Train scores (lemmas) :::')
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(predictions=train_preds['lemma_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.train_lemmas,
                                                 silver=pred_lemmas,
                                                 test_tokens=self.train_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
            """
            print('::: Dev scores (lemmas) :::')
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(predictions=dev_preds['lemma_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.dev_lemmas,
                                                 silver=pred_lemmas,
                                                 test_tokens=self.dev_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
            """

        if self.include_pos:
            print('::: Train scores (pos) :::')
            pred_pos = self.preprocessor.inverse_transform_pos(predictions=train_preds['pos_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.train_pos,
                                                 silver=pred_pos,
                                                 test_tokens=self.train_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
            """
            print('::: Dev scores (pos) :::')
            pred_pos = self.preprocessor.inverse_transform_pos(predictions=dev_preds['pos_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.dev_pos,
                                                 silver=pred_pos,
                                                 test_tokens=self.dev_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
            """
        
        if self.include_morph:
            print('::: Dev scores (morph) :::')
            """
            pred_morph = self.preprocessor.inverse_transform_morph(predictions=dev_preds['morph_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.dev_morph,
                                                 silver=pred_morph,
                                                 test_tokens=self.dev_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
            """

        


                
    
        

        
        
