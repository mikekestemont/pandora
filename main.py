#!usr/bin/env python
# -*- coding: utf-8 -*-

import pandora.utils
from pandora.tagger import Tagger

import os, codecs

def main():
    print('::: started :::')
    num_epochs = 30

    # set hyperparameters:
    nb_encoding_layers = 1
    nb_dense_dims = 300
    batch_size = 100
    nb_left_tokens = 2
    nb_right_tokens = 2
    nb_embedding_dims = 150
    model_name = 'full'
    postcorrect = True
    include_token = True
    include_context = True
    include_lemma = 'generate' # or None, or 'generate'
    include_pos = True
    include_morph = True
    include_dev = False
    include_test = True
    nb_filters = 100
    filter_length = 3
    focus_repr = 'recurrent'
    dropout_level = .15

    train_data = pandora.utils.load_annotated_file('data/capitula_classic/train0.tsv',
                                            format='tab',
                                            include_pos=include_pos,
                                            include_lemma=include_lemma,
                                            include_morph=include_morph,
                                            nb_instances=None)
    test_data = pandora.utils.load_annotated_file('data/capitula_classic/test0.tsv',
                                            format='tab',
                                            include_pos=include_pos,
                                            include_lemma=include_lemma,
                                            include_morph=include_morph,
                                            nb_instances=None)
    
    tagger = Tagger(nb_encoding_layers = nb_encoding_layers,
                    nb_dense_dims = nb_dense_dims,
                    batch_size = batch_size,
                    nb_left_tokens = nb_left_tokens,
                    nb_right_tokens = nb_right_tokens,
                    nb_embedding_dims = nb_embedding_dims,
                    model_name = model_name,
                    postcorrect = postcorrect,
                    include_token = include_token,
                    include_context = include_context,
                    include_lemma = include_lemma,
                    include_pos = include_pos,
                    include_morph = include_morph,
                    include_dev = include_dev,
                    include_test = include_test,
                    nb_filters = nb_filters,
                    filter_length = filter_length,
                    focus_repr = focus_repr,
                    dropout_level = dropout_level,
                    )

    tagger.setup_to_train(train_data=train_data,
                          test_data=test_data)

    for i in range(num_epochs):
        tagger.epoch()
        tagger.test()

    """
    tagger.save()

    tagger = Tagger(nb_encoding_layers = nb_encoding_layers,
                    nb_dense_dims = nb_dense_dims,
                    batch_size = batch_size,
                    nb_left_tokens = nb_left_tokens,
                    nb_right_tokens = nb_right_tokens,
                    nb_embedding_dims = nb_embedding_dims,
                    model_name = model_name,
                    postcorrect = postcorrect,
                    include_token = include_token,
                    include_context = include_context,
                    include_lemma = include_lemma,
                    include_pos = include_pos,
                    include_morph = include_morph,
                    include_dev = include_dev,
                    include_test = include_test,
                    nb_filters = nb_filters,
                    filter_length = filter_length,
                    focus_repr = focus_repr,
                    dropout_level = dropout_level,
                    )
    tagger.load()
    """

    print('annotating...')
    orig_path = 'data/12C/orig/'
    new_path = 'data/12C/tagged/'
    for filename in os.listdir(orig_path):
        if not filename.endswith('.txt'):
            continue
        unseen_tokens = pandora.utils.load_unannotated_file(orig_path + filename,
                                                         nb_instances=None,
                                                         tokenized_input=False)
        annotations = tagger.annotate(unseen_tokens)
        with codecs.open(new_path + filename, 'w', 'utf8') as f:
            for t, l, p, m in zip(annotations['tokens'], annotations['lemmas'], annotations['pos'], annotations['morph']):
                f.write(' '.join((t, l, p, m))+'\n')
    
    print('::: ended :::')

if __name__ == '__main__':
    main()

