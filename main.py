#!usr/bin/env python
# -*- coding: utf-8 -*-

import pandora.utils
import pandora.evaluation
from pandora.tagger import Tagger

def main():
    print('::: started :::')
    train_insts = pandora.utils.load_annotated_data(directory='data/capitula0/',
                                            format='conll',
                                            nb_instances=30000)
    dev_insts = pandora.utils.load_annotated_data(directory='data/capitula1/',
                                            format='conll',
                                            nb_instances=5000)
    
    tagger = Tagger()
    tagger.fit(train_instances=train_insts,
               dev_instances=dev_insts)
    
    print('::: ended :::')

if __name__ == '__main__':
    main()