#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from operator import itemgetter
import numpy as np

def single_label_accuracies(gold, silver, test_tokens, known_tokens):
    """
    Calculate accuracies for all, known and unknown tokens.
    Uses index of items seen during training.
    """
    kno_corr, unk_corr = 0.0, 0.0
    nb_kno, nb_unk = 0.0, 0.0

    for gold_pred, silver_pred, tok in zip(gold, silver, test_tokens):

        if tok in known_tokens:
            nb_kno += 1
            if gold_pred == silver_pred:
                kno_corr += 1
        else:
            nb_unk += 1
            if gold_pred == silver_pred:
                unk_corr += 1

    all_acc = (kno_corr + unk_corr) / (nb_kno + nb_unk)
    kno_acc = kno_corr / nb_kno

    # account for situation with no unknowns:
    unk_acc = 1.0
    if nb_unk > 0:
        unk_acc = unk_corr / nb_unk

    return all_acc, kno_acc, unk_acc

def multilabel_accuracies(gold, silver, test_tokens, known_tokens):
    """
    Calculate accuracies for all, known and unknown tokens.
    Uses index of items seen during training.
    """
    kno_corr, unk_corr = 0.0, 0.0
    nb_kno, nb_unk = 0.0, 0.0

    for gold_pred, silver_pred, tok in zip(gold, silver, test_tokens):

        if tok in known_tokens:
            nb_kno += 1
            if set(gold_pred) == set(silver_pred):
                kno_corr += 1
        else:
            nb_unk += 1
            if set(gold_pred) == set(silver_pred):
                unk_corr += 1

    all_acc = (kno_corr + unk_corr) / (nb_kno + nb_unk)
    kno_acc = kno_corr / nb_kno

    # account for situation with no unknowns:
    unk_acc = 1.0
    if nb_unk > 0:
        unk_acc = unk_corr / nb_unk

    return all_acc, kno_acc, unk_acc


    