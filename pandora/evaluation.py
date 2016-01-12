#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from operator import itemgetter
import numpy as np

def convert_to_lemmas(predictions, out_char_idx):
    """
    For each prediction, convert 2D char matrix
    with probabilities to actual lemmas, using
    character index for the output strings.
    """
    pred_lemmas = []
    for pred in predictions:
        pred_lem = ''
        for positions in pred:
            top_idx = np.argmax(positions) # winning position
            c = out_char_idx[top_idx] # look up corresponding char
            pred_lem += c # add character
        pred_lemmas.append(pred_lem)
    return pred_lemmas

def accuracies(gold, silver, test_tokens, known_tokens):
    """
    Calculate accuracies for all, known and unknown tokens.
    Uses index of items seen during training.
    """
    kno_corr, unk_corr = 0.0, 0.0
    nb_kno, nb_unk = 0.0, 0.0

    for gold_lem, silver_lem, tok in zip(gold, silver, test_tokens):
        # rm trailing $:
        try:
            silver_lem = silver_lem[:silver_lem.index('$')]
        except ValueError:
            pass

        if tok in known_tokens:
            nb_kno += 1
            if gold_lem == silver_lem:
                kno_corr += 1
        else:
            nb_unk += 1
            if gold_lem == silver_lem:
                unk_corr += 1

    all_acc = (kno_corr + unk_corr) / (nb_kno + nb_unk)
    kno_acc = kno_corr / nb_kno

    # account for situation with no unknowns:
    unk_acc = 1.0
    if nb_unk > 0:
        unk_acc = unk_corr / nb_unk

    return all_acc, kno_acc, unk_acc


    