#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import Counter
from operator import itemgetter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE


class SentenceIterator:
    def __init__(self, tokens, sentence_len=100):
        
        self.sentence_len = sentence_len
        self.tokens = [t.lower() for t in tokens]
        self.idxs = []
        start_idx, end_idx = 0, self.sentence_len
        while end_idx < len(self.tokens):
            self.idxs.append((start_idx, end_idx))
            start_idx += self.sentence_len
            end_idx += self.sentence_len

    def __iter__(self):
        for start_idx, end_idx in self.idxs:
            yield self.tokens[start_idx : end_idx]


class Pretrainer:

    def __init__(self, nb_left_tokens, nb_right_tokens,
                 sentence_len=100, window=5,
                 min_count=1, size=300, nb_mfi=500,
                 nb_workers=10, nb_negative=5,
                 ):
        self.nb_left_tokens = nb_left_tokens
        self.nb_right_tokens = nb_right_tokens
        self.size = size
        self.nb_mfi = nb_mfi
        self.window = window
        self.min_count = min_count
        self.nb_workers = nb_workers
        self.nb_negative = nb_negative

    def fit(self, tokens):
        # get most frequent items for plotting:
        tokens = [t.lower() for t in tokens]
        self.mfi = [t for t,_ in Counter(tokens).most_common(self.nb_mfi)]
        self.sentence_iterator = SentenceIterator(tokens=tokens)
        # train embeddings:
        self.w2v_model = Word2Vec(self.sentence_iterator,
                             window=self.window,
                             min_count=self.min_count,
                             size=self.size,
                             workers=self.nb_workers,
                             negative=self.nb_negative)
        self.plot_mfi()
        self.most_similar()

        # build an index of the train tokens
        # which occur at least min_count times:
        self.token_idx = {'<unk>': 0}
        for k, v in Counter(tokens).items():
            if v >= self.min_count:
                self.token_idx[k] = len(self.token_idx)

        # create an ordered vocab:
        self.train_token_vocab = [k for k, v in sorted(self.token_idx.items(),\
                        key=itemgetter(1))]
        self.pretrained_embeddings = self.get_weights(self.train_token_vocab)

        return self

    def get_weights(self, vocab):
        unk = np.zeros(self.size)
        weights = []
        for w in vocab:
            try:
                weights.append(self.w2v_model[w])
            except KeyError:
                weights.append(unk)
        return [np.asarray(weights)]

    def transform(self, tokens):

        context_ints = []
        tokens = [t.lower() for t in tokens]
        for curr_idx, token in enumerate(tokens):
            ints = []
            # vectorize left context:
            left_context_tokens = [tokens[curr_idx-(t+1)]\
                                    for t in range(self.nb_left_tokens)\
                                        if curr_idx-(t+1) >= 0][::-1]
            idxs = []
            if left_context_tokens:
                idxs = [self.token_idx[t] if t in self.token_idx else 0 \
                            for t in left_context_tokens]
            while len(idxs) < self.nb_left_tokens:
                idxs = [0] + idxs
            ints.extend(idxs)

            # vectorize right context
            right_context_tokens = [tokens[curr_idx+(t+1)]\
                                        for t in range(self.nb_right_tokens)\
                                            if curr_idx+(t+1) < len(tokens)]
            idxs = []
            if right_context_tokens:
                idxs = [self.token_idx[t] if t in self.token_idx else 0 \
                            for t in right_context_tokens]
            while len(idxs) < self.nb_right_tokens:
                idxs.append(0)
            ints.extend(idxs)

            context_ints.append(ints)

        return np.asarray(context_ints, dtype='int8')


    def plot_mfi(self, outputfile='embeddings.pdf', nb_clusters=8, weights='NA'):
        # collect embeddings for mfi:
        X = np.asarray([self.w2v_model[w] for w in self.mfi \
                            if w in self.w2v_model], dtype='float32')
        # dimension reduction:
        tsne = TSNE(n_components=2)
        coor = tsne.fit_transform(X) # unsparsify

        plt.clf()
        sns.set_style('dark')
        sns.plt.rcParams['axes.linewidth'] = 0.4
        fig, ax1 = sns.plt.subplots()  

        labels = self.mfi
        # first plot slices:
        x1, x2 = coor[:,0], coor[:,1]
        ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
        # clustering on top (add some colouring):
        clustering = AgglomerativeClustering(linkage='ward',
                            affinity='euclidean', n_clusters=nb_clusters)
        clustering.fit(coor)
        # add names:
        for x, y, name, cluster_label in zip(x1, x2, labels, clustering.labels_):
            ax1.text(x, y, name, ha='center', va="center",
                     color=plt.cm.spectral(cluster_label / 10.),
                     fontdict={'family': 'Arial', 'size': 8})
        # control aesthetics:
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_yticklabels([])
        ax1.set_yticks([])
        sns.plt.savefig(outputfile, bbox_inches=0)

    def most_similar(self, nb_neighbors=5,
                     words=['doet', 'goet', 'ende', 'mach', 'gode'],
                     outputfile='neighbours.txt'):
        with open(outputfile, 'w') as f:
            for w in words:
                try:
                    neighbors = ' - '.join([v for v,_ in self.w2v_model.most_similar(w)])
                    f.write(' '.join((w, '>', neighbors))+'\n')
                    f.write(':::::::::::::::::\n')
                except KeyError:
                    pass
