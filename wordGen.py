# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:00:24 2023

@author: zfoong
"""

import random
from nltk.corpus import wordnet, brown
import nltk
import os
import json


class wordGen:
    
    def __init__(self, mode='simple', seed=None):
        self.mode = 'simple' # 'full', 'simple'
        if seed is not None:
            random.seed(seed)
            
        # get a list of the most common 10,000 words in the english
        self.common_words = set([word.lower() for word in brown.words() if word.isalpha()][:10000])
        
        # get a list of all the noun synsets in WordNet
        self.noun_synsets = list(wordnet.all_synsets('n'))
    
    def get_word(self, count):
        nouns = []
        word_count = count
        
        
        # generate 10 random easy nouns
        while len(nouns) < word_count:
            # select a random synset from the list of noun synsets
            noun_synset = random.choice(self.noun_synsets)
            # get the first lemma from the synset
            noun = noun_synset.lemmas()[0].name()
            # check if the noun is a simple noun and is among the most common words
            
            # if '_' not in noun and noun[0].islower() and noun in common_words:
            if '_' in noun:
                continue 
            
            # check duplicate
            if len(nouns) != len(set(nouns)):
                nouns = []
                continue
            
            if self.mode == 'full':
                pass
            if self.mode == 'simple':
                if noun not in self.common_words:
                    continue
            else:
                raise Exception("mode not found")
                
            nouns.append(noun)
        
        # print(nouns)
        return nouns

if __name__ == "__main__":
    
    folder_name = "word_list"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    nltk.download('brown')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    wg = wordGen('simple')
    word_gen_counts = 10 # N
    word_lists = []
    
    for i in range(word_gen_counts):
        word_list = wg.get_word(5)
        word_lists.append(word_list)
        
    print(word_lists)
    
    file_path = os.path.join(folder_name, 'words_test.txt')
    with open(file_path, 'w') as file:
        json.dump(word_lists, file)
        
    # with open(file_path, 'r') as file:
        # loaded_word_list = json.load(file)