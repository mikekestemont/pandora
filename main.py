#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys

import pandora.utils
from pandora.tagger import Tagger

import os, codecs

def main():
    print('::: started :::')
    cf_path = sys.argv[1]
    params = pandora.utils.get_param_dict(cf_path)
    params['config_path'] = cf_path
    
    train_data = pandora.utils.load_annotated_file('data/mdu/cg-lit/cg-lit_train.tab',
    #train_data = pandora.utils.load_annotated_dir('data/mdu/all_train',
                                            format='tab',
    #                                        extension='.tab',
                                            include_pos=params['include_pos'],
                                            include_lemma=params['include_lemma'],
                                            include_morph=params['include_morph'],
                                            nb_instances=None)
    dev_data = pandora.utils.load_annotated_file('data/mdu/cg-lit/cg-lit_dev.tab',
    #dev_data = pandora.utils.load_annotated_dir('data/mdu/all_test',
                                            format='tab',
    #                                        extension='.tab',
                                            include_pos=params['include_pos'],
                                            include_lemma=params['include_lemma'],
                                            include_morph=params['include_morph'],
                                            nb_instances=None)

    #test_data = pandora.utils.load_annotated_file('data/capitula_classic/test0.tsv',
    #dev_data = pandora.utils.load_annotated_file('data/mdu/cg-lit/cg-lit_dev.tab',
    #dev_data = pandora.utils.load_annotated_file('data/EMDu/train.txt',
    #test_data = pandora.utils.load_annotated_file('data/mdu/cg-lit/cg-lit_test.tab',
    #                                        format='tab',
    #                                        include_pos=params['include_pos'],
    #                                        include_lemma=params['include_lemma'],
    #                                        include_morph=params['include_morph'],
    #                                        nb_instances=None)

    
    tagger = Tagger(**params)
    tagger.setup_to_train(train_data=train_data,
                          dev_data=dev_data)

    for i in range(int(params['nb_epochs'])):
        tagger.epoch()
        tagger.save()

    tagger.save()
    #tagger = Tagger(load=True, model_dir='models/mdu_all')

    
    
    print('::: ended :::')

if __name__ == '__main__':
    main()

