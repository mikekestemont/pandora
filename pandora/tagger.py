#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import cPickle as pickle
import os
import codecs

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


import utils
import evaluation
from model import build_model
from preprocessing import Preprocessor
from pretraining import Pretrainer

MODELS_DIR = 'models'
if not os.path.isdir(MODELS_DIR):
    os.mkdir(MODELS_DIR)

class Tagger():

    def __init__(self):
        # set hyperparameters:
        self.nb_encoding_layers = 3
        self.nb_dense_dims = 150
        self.batch_size = 150
        self.nb_left_tokens = 3
        self.nb_right_tokens = 3
        self.nb_context_tokens = self.nb_left_tokens + self.nb_right_tokens
        self.nb_embedding_dims = 100
        self.model_name = 'XXX'

        # which subnets?
        self.include_token = True
        self.include_context = True

        # which headnets?
        self.include_lemma = True
        self.include_pos = True
        self.include_morph = False

        self.complex_pos = False

        # initialize:
        self.setup = False
        self.nb_epochs = 0

    def setup_(self, train_instances, test_instances, load_pickles=False):
        self.model_path = os.sep.join((MODELS_DIR, self.model_name))
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

        # load test data:
        self.test_tokens, self.test_lemmas, self.test_pos, self.test_morph = \
            zip(*test_instances)

        # load train data and split off dev set:
        b = int(len(train_instances) * 0.9)
        self.train_tokens, self.train_lemmas, self.train_pos, self.train_morph = \
            zip(*train_instances[:b])
        self.dev_tokens, self.dev_lemmas, self.dev_pos, self.dev_morph = \
            zip(*train_instances[b:])

        if self.complex_pos:
            self.train_pos = tuple('-'.join(i) for i in zip(self.train_pos, self.train_morph))
            self.dev_pos = tuple('-'.join(i) for i in zip(self.dev_pos, self.dev_morph))
            self.test_pos = tuple('-'.join(i) for i in zip(self.test_pos, self.test_morph))

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

        self.train_X_focus, self.train_X_lemma, self.train_X_pos, self.train_X_morph = \
            self.preprocessor.transform(tokens=self.train_tokens,
                                               lemmas=self.train_lemmas,
                                               pos=self.train_pos,
                                               morph=self.train_morph)

        self.dev_X_focus, self.dev_X_lemma, self.dev_X_pos, self.dev_X_morph = \
            self.preprocessor.transform(tokens=self.dev_tokens,
                                        lemmas=self.dev_lemmas,
                                        pos=self.dev_pos,
                                        morph=self.dev_morph)

        self.test_X_focus, self.test_X_lemma, self.test_X_pos, self.test_X_morph = \
            self.preprocessor.transform(tokens=self.test_tokens,
                                        lemmas=self.test_lemmas,
                                        pos=self.test_pos,
                                        morph=self.test_morph)

        self.train_contexts = self.pretrainer.transform(tokens=self.train_tokens)
        self.dev_contexts = self.pretrainer.transform(tokens=self.dev_tokens)
        self.test_contexts = self.pretrainer.transform(tokens=self.test_tokens)

        
        if load_pickles:
            print('Re-building model...')
            self.model = model_from_json(open(os.sep.join((self.model_path, 'model_architecture.json'))).read())
            self.model.load_weights(os.sep.join((self.model_path, 'model_weights.hdf5')))
        else:
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
        if self.nb_epochs % 15 == 0:
            old_lr  = self.model.optimizer.lr.get_value()
            new_lr = np.float32(old_lr * 0.33)
            self.model.optimizer.lr.set_value(new_lr)
            print('\t- Lowering learning rate > was:', old_lr, ', now:', new_lr)

        if self.nb_epochs % 5 == 0 and self.batch_size >= 25:
            new_batch_size = int(self.batch_size * 0.8)
            print('\t- Lowering batch_size > was:', self.batch_size, ', now:', new_batch_size)
            self.batch_size = new_batch_size
        
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
              shuffle = True,
              batch_size = self.batch_size)

        # get train loss:
        train_loss = self.model.evaluate(data=d,
                                batch_size=self.batch_size)
        print("\t - total train loss:\t{:.3}".format(train_loss))

        # get train preds:
        train_preds = self.model.predict(data=d,
                                batch_size=self.batch_size)

        # get dev predictions:
        d = {}
        if self.include_token:
            d['focus_in'] = self.dev_X_focus
        if self.include_context:
            d['context_in'] = self.dev_contexts

        dev_preds = self.model.predict(data=d,
                                batch_size=self.batch_size)

        if self.include_lemma:
            print('::: Train scores (lemmas) :::')
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(predictions=train_preds['lemma_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.train_lemmas,
                                                 silver=pred_lemmas,
                                                 test_tokens=self.train_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
            print('::: Dev scores (lemmas) :::')
            pred_lemmas = self.preprocessor.inverse_transform_lemmas(predictions=dev_preds['lemma_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.dev_lemmas,
                                                 silver=pred_lemmas,
                                                 test_tokens=self.dev_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)

        if self.include_pos:
            print('::: Train scores (pos) :::')
            pred_pos = self.preprocessor.inverse_transform_pos(predictions=train_preds['pos_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.train_pos,
                                                 silver=pred_pos,
                                                 test_tokens=self.train_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
            print('::: Dev scores (pos) :::')
            pred_pos = self.preprocessor.inverse_transform_pos(predictions=dev_preds['pos_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.dev_pos,
                                                 silver=pred_pos,
                                                 test_tokens=self.dev_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)
        
        if self.include_morph:     
            print('::: Dev scores (morph) :::')
            pred_morph = self.preprocessor.inverse_transform_morph(predictions=dev_preds['morph_out'])
            all_acc, kno_acc, unk_acc = evaluation.single_label_accuracies(gold=self.dev_morph,
                                                 silver=pred_morph,
                                                 test_tokens=self.dev_tokens,
                                                 known_tokens=self.preprocessor.known_tokens)

        print('><><><><><><><><><><><><><><><><><><><><><><><><><><')

        ##### OUT #######################################################################################
        if self.include_lemma and not self.include_pos:
            with codecs.open(os.sep.join((self.model_path, 'dev_out.txt')), 'w', 'utf8') as f:
                for p in zip(self.dev_tokens, self.dev_lemmas, pred_lemmas):
                    try:
                        f.write('\t'.join([str(r) for r in p]) + '\n')
                    except:
                        pass
        elif not self.include_lemma and self.include_pos:
            with codecs.open(os.sep.join((self.model_path, 'dev_out.txt')), 'w', 'utf8') as f:
                for p in zip(self.dev_tokens, self.dev_pos, pred_pos):
                    try:
                        f.write('\t'.join([str(r) for r in p]) + '\n')
                    except:
                        pass
        elif self.include_lemma and self.include_pos:
            with codecs.open(os.sep.join((self.model_path, 'dev_out.txt')), 'w', 'utf8') as f:
                for p in zip(self.dev_tokens, self.dev_pos, pred_pos, self.dev_lemmas, pred_lemmas):
                    try:
                        f.write('\t'.join([str(r) for r in p]) + '\n')
                    except:
                        pass

        self.save()




        


                
    
        

        
        
