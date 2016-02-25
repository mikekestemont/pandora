#!usr/bin/env python
# -*- coding: utf-8 -*-

import pandora.utils
import pandora.evaluation
from pandora.tagger import Tagger

def main():
    print('::: started :::')
    train_insts = pandora.utils.load_annotated_file('data/latin/train.conll',
                                            format='conll',
                                            nb_instances=100000000000)
    test_insts = pandora.utils.load_annotated_file('data/latin/test.conll',
                                            format='conll',
                                            nb_instances=100000000000)
    
    tagger = Tagger()
    tagger.setup_(train_instances=train_insts,
                  test_instances=test_insts,
                  load_pickles=False)
    
    for i in range(50):
        tagger.epoch()
        tagger.test()
    
    print('::: ended :::')

if __name__ == '__main__':
    main()