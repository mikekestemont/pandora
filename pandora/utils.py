#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import codecs
import re
import ConfigParser

def load_annotated_data(directory='directory', format='conll', nb_instances=None,
                        include_lemma=True, include_morph=True, include_pos=True):
    instances = []
    for filepath in glob.glob(directory+'/*'):
        insts = load_annotated_file(filepath=filepath,
                                    format=format,
                                    nb_instances=nb_instances,
                                    include_lemma=include_lemma,
                                    include_morph=include_morph,
                                    include_pos=include_pos)
        instances.extend(insts)
    return instances

def load_annotated_file(filepath='text.txt', format='tab', nb_instances=None,
                        include_lemma=True, include_morph=True,
                        include_pos=True):
    instances = {'token': []}
    if include_lemma:
        instances['lemma'] = []
    if include_pos:
        instances['pos'] = []
    if include_morph:
        instances['morph'] = []
    if format == 'conll':
        for line in codecs.open(filepath, 'r', 'utf8'):
            line = line.strip()
            if line:
                try:
                    idx, tok, _, lem, _, pos, morph = \
                        line.split()[:7]
                    if include_lemma:
                        lem = lem.lower()
                    if include_morph:
                        '|'.join(sorted(set(morph.split('|'))))
                    
                    instances['token'].append(tok)
                    if include_lemma:
                        instances['lemma'].append(lem)
                    if include_pos:
                        instances['pos'].append(pos)
                    if include_morph:
                        instances['morph'].append(morph)
                except ValueError:
                    pass
            if nb_instances:
                if len(instances) >= nb_instances:
                    break
    elif format == 'tab':
        for line in codecs.open(filepath, 'r', 'utf8'):
            line = line.strip()
            if line and not line[0] == '@':
                try:
                    comps = line.split()
                    tok = comps[0]
                    if include_lemma:
                        lem = comps[1].lower()
                    if include_pos:
                        pos = comps[2]
                    if include_morph:
                        morph = '|'.join(sorted(set(comps[3].split('|'))))
                    
                    instances['token'].append(tok)
                    if include_lemma:
                        instances['lemma'].append(lem)
                    if include_pos:
                        instances['pos'].append(pos)
                    if include_morph:
                        instances['morph'].append(morph)
                except ValueError:
                    pass
            if nb_instances:
                if len(instances['token']) >= nb_instances:
                    break
    return instances

def load_unannotated_file(filepath='test.txt', nb_instances=None, tokenized_input=False):
    if tokenized_input:
        instances = []
        for line in codecs.open(filepath, 'r', 'utf8'):
            line = line.strip()
            if line:
                instances.append(line)
            if nb_instances:
                nb_instances -= 1
                if nb_instances <= 0:
                    break
        return instances
    else:
        from nltk.tokenize import wordpunct_tokenize
        W = re.compile('\s+')
        with codecs.open(filepath, 'r', 'utf8') as f:
            text = W.sub(f.read(), ' ')
        tokens = wordpunct_tokenize(text)
        if nb_instances:
            return tokens[:nb_instances]
        else:
            return tokens

def stats(tokens, lemmas, known):
    print('Nb of tokens:', len(tokens))
    print('Nb of unique tokens:', len(set(tokens)))
    cnt = sum([1.0 for k in tokens if k not in known])/len(tokens)
    cnt *= 100.0
    print('Nb of unseen tokens:', cnt)
    print('Nb of unique lemmas: ', len(set(lemmas)))


def get_param_dict(p):
    config = ConfigParser.ConfigParser()
    config.read(p)
    # parse the param
    param_dict = dict()
    for section in config.sections():
        for name, value in config.items(section):
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            param_dict[name] = value
    return param_dict


