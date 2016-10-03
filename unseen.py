#!usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys

import pandora.utils
from pandora.tagger import Tagger

import os, codecs

def main():
    print('::: started :::')
    
    tagger = Tagger(load=True, model_dir='models/wilhelmus_full')

    print('Tagger loaded, now annotating...')

    orig_path = 'data/wilhelmus/orig/'
    new_path = 'data/wilhelmus/tagged/'

    for filename in os.listdir(orig_path):
        if not filename.endswith('.txt'):
            continue

        print('\t +', filename)
        unseen_tokens = pandora.utils.load_unannotated_file(orig_path + filename,
                                                         nb_instances=None,
                                                         tokenized_input=False)

        annotations = tagger.annotate(unseen_tokens)
        with codecs.open(new_path + filename, 'w', 'utf8') as f:
            #for t, l, p in zip(annotations['tokens'], annotations['postcorrect_lemmas'], annotations['pos']):
            for t, l, p in zip(annotations['tokens'], annotations['lemmas'], annotations['pos']):
            #for t, l in zip(annotations['tokens'], annotations['lemmas']):
                f.write('\t'.join((t, l, p))+'\n')
    
    print('::: ended :::')

if __name__ == '__main__':
    main()

