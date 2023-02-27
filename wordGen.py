# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:00:24 2023

@author: zfoong
"""

import random
from nltk.corpus import wordnet, brown

class wordGen:
    
    def __init__(self, mode='simple', seed=None):
        self.mode = 'simple' # 'full', 'simple'
        if seed is not None:
            random.seed(seed)
    
    def get_word(self, count):
        nouns = []
        word_count = count
        
        # get a list of the most common 10,000 words in the english
        common_words = set([word.lower() for word in brown.words() if word.isalpha()][:10000])
        
        # get a list of all the noun synsets in WordNet
        noun_synsets = list(wordnet.all_synsets('n'))
        
        # generate 10 random easy nouns
        while len(nouns) < word_count:
            # select a random synset from the list of noun synsets
            noun_synset = random.choice(noun_synsets)
            # get the first lemma from the synset
            noun = noun_synset.lemmas()[0].name()
            # check if the noun is a simple noun and is among the most common words
            
            # if '_' not in noun and noun[0].islower() and noun in common_words:
            if '_' in noun:
                continue 
            
            if self.mode == 'full':
                pass
            if self.mode == 'simple':
                if noun not in common_words:
                    continue
            else:
                raise Exception("mode not found")
                
            nouns.append(noun)
        
        print(nouns)
        return nouns

if __name__ == "__main__":
    wg = wordGen('simple')
    wg.get_word(10)