#!usr/bin/env python
# -*- coding: utf-8 -*-

import pandora.utils
from pandora.tagger import Tagger

import os, codecs

def main():
    print('::: started :::')
    
    params = pandora.utils.get_param_dict('config.txt')

    train_data = pandora.utils.load_annotated_file('data/capitula_classic/train0.tsv',
    #train_data = pandora.utils.load_annotated_file('data/mdu/cg-lit/cg-lit_train.tab',
                                            format='tab',
                                            include_pos=params['include_pos'],
                                            include_lemma=params['include_lemma'],
                                            include_morph=params['include_morph'],
                                            nb_instances=10000)
    test_data = pandora.utils.load_annotated_file('data/capitula_classic/test0.tsv',
    #dev_data = pandora.utils.load_annotated_file('data/mdu/cg-lit/cg-lit_dev.tab',
                                            format='tab',
                                            include_pos=params['include_pos'],
                                            include_lemma=params['include_lemma'],
                                            include_morph=params['include_morph'],
                                            nb_instances=100)
    
    tagger = Tagger(**params)
    tagger.setup_to_train(train_data=train_data,
                          dev_data=dev_data)

    tagger.train()
    
    tagger.test()
    tagger.save()
    
    tagger = Tagger(load=True, model_dir='models/full')

    print('annotating...')
    orig_path = 'data/12C/orig/'
    new_path = 'data/12C/tagged/'
    for filename in os.listdir(orig_path):
        if not filename.endswith('.txt'):
            continue
        unseen_tokens = pandora.utils.load_unannotated_file(orig_path + filename,
                                                         nb_instances=100,
                                                         tokenized_input=False)
        annotations = tagger.annotate(unseen_tokens)
        with codecs.open(new_path + filename, 'w', 'utf8') as f:
            if postcorrect:
                for t, l, pl, p, m in zip(annotations['tokens'], annotations['lemmas'], annotations['postcorrect_lemmas'], annotations['pos'], annotations['morph']):
                    f.write(' '.join((t, l, pl, p, m))+'\n')
            else:
                for t, l, p, m in zip(annotations['tokens'], annotations['lemmas'], annotations['pos'], annotations['morph']):
                    f.write(' '.join((t, l, p, m))+'\n')
    
    
    print('::: ended :::')

if __name__ == '__main__':
    main()

