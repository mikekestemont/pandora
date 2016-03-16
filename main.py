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
    nb_embedding_dims = 100
    model_name = 'new'
    postcorrect = True
    include_token = True
    include_context = True
    include_lemma = True
    include_pos = True
    include_morph = True
    complex_pos = False

    train_data = pandora.utils.load_annotated_file('data/capitula/train0.tsv',
                                            format='tab',
                                            include_pos=include_pos,
                                            include_lemma=include_lemma,
                                            include_morph=include_morph,
                                            nb_instances=5000)
    test_data = pandora.utils.load_annotated_file('data/capitula/test0.tsv',
                                            format='tab',
                                            include_pos=include_pos,
                                            include_lemma=include_lemma,
                                            include_morph=include_morph,
                                            nb_instances=1000)
    
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
                    complex_pos = complex_pos)

    for items in train_data:
        print(len(train_data[items]))
    tagger.setup_(train_data=train_data,
                  test_data=test_data,
                  load_pickles=False)

    for i in range(100):
        tagger.epoch()
        tagger.test()
    
    print('::: ended :::')

if __name__ == '__main__':
    main()

