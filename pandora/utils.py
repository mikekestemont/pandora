#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import codecs

import numpy as np

def load_annotated_data(directory, format='conll', nb_instances=None,
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

def load_annotated_file(filepath, format, nb_instances=None,
                        include_lemma=True, include_morph=True, include_pos=True):
    instances = []
    if format == 'conll':
        for line in codecs.open(filepath, 'r', 'utf8'):
            line = line.strip()
            if line:
                try:
                    idx, tok, _, lem, _, pos, morph = \
                        line.split()[:7]
                    tok = tok.lower()
                    lem = lem.lower()
                    inst = [tok]
                    if include_lemma:
                        inst.append(lem)
                    if include_pos:
                        inst.append(pos)
                    if include_morph:
                        morph = morph.split('|')
                        morph = '+'.join(sorted(set(morph)))
                        inst.append(morph)
                    instances.append(tuple(inst))
                except ValueError:
                    pass
            if nb_instances:
                if len(instances) >= nb_instances:
                    break
    return instances

def load_raw_file(filepath, nb_instances=1000):
    instances = []
    for line in codecs.open(filepath, 'r', 'utf8'):
        line = line.strip()
        if line:
            instances.append(line)
        nb_instances -= 1
        if nb_instances <= 0:
            break
    return instances