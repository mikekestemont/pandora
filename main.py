#!usr/bin/env python
# -*- coding: utf-8 -*-

import pandora.utils
import pandora.evaluation
from pandora.tagger import Tagger

def main():
    print('::: started :::')

    # set hyperparameters:
    nb_encoding_layers = 1
    nb_dense_dims = 300
    batch_size = 100
    nb_left_tokens = 2
    nb_right_tokens = 2
    nb_embedding_dims = 150
    model_name = 'new'
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
    dropout_level = .1

    train_data = pandora.utils.load_annotated_file('data/capitula/train0.tsv',
                                            format='tab',
                                            include_pos=include_pos,
                                            include_lemma=include_lemma,
                                            include_morph=include_morph,
                                            nb_instances=10000000000000000)
    test_data = pandora.utils.load_annotated_file('data/capitula/test0.tsv',
                                            format='tab',
                                            include_pos=include_pos,
                                            include_lemma=include_lemma,
                                            include_morph=include_morph,
                                            nb_instances=10000000000000000)
    
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

    tagger.setup_(train_data=train_data,
                  test_data=test_data,
                  load_pickles=False)

    for i in range(100):
        tagger.epoch()
        #tagger.test()
    
    print('::: ended :::')

if __name__ == '__main__':
    main()

