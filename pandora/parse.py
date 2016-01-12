#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup

def main():
    print('::: started :::')

    with open('../data/Perseus_lexicon.xml', 'rb') as infile:
        tree = BeautifulSoup(infile)
    
    for lemma_node in tree.findAll('entryfree'):
        print(lemma_node['key'])
    
    print('::: ended :::')

if __name__ == '__main__':
    main()