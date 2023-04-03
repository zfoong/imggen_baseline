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
import numpy as np

def add_article(word):
    vowels = ['a', 'e', 'i', 'o', 'u']
    if word[0].lower() in vowels:
        return 'an ' + word
    else:
        return 'a ' + word

class wordGen:
    
    def __init__(self, mode='simple', seed=None):
        self.mode = 'simple' # 'full', 'simple'
        if seed is not None:
            random.seed(seed)
            
        # get a list of the most common 10,000 words in the english
        self.common_words = set([word.lower() for word in brown.words() if word.isalpha()][:10000])
        
        # get a list of all the noun synsets in WordNet
        self.noun_synsets = list(wordnet.all_synsets('n'))
        
                
        with open("template_prompts.txt", "r") as file:
            template_prompts = file.readlines()
        self.template_prompts = [prompt.strip() for prompt in template_prompts]
        
        with open("imagenet_label.txt", "r") as file:
            classes = file.readlines()
        self.classes = [prompt.strip() for prompt in classes]
    
    def get_word_old(self, count):
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
    
    def get_word(self, count):
        nouns = []
        word_count = count

        template = random.choice(self.template_prompts)

        # generate 10 random easy nouns
        while len(nouns) < word_count:

            noun = random.choice(self.classes)
            
            # check duplicate
            if len(nouns) != len(set(nouns)):
                nouns = []
                continue
                
            nouns.append(noun)
        
        # print(nouns)
        formatted_nouns = [add_article(n) for n in nouns]
        
        prompts = None
        if len(formatted_nouns) > 1:
            last_item = formatted_nouns.pop()
            prompts = template.format(', '.join(formatted_nouns) + ', and ' + last_item)
        else:
            prompts = template.format(formatted_nouns[0])
        
        return prompts, nouns
    

if __name__ == "__main__":
    
    folder_name = "word_list"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    nltk.download('brown')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    wg = wordGen('simple')
    word_gen_counts = 10 # N
    prompt_lists = []
    noun_lists = []
    
    for i in range(word_gen_counts):
        prompt_list, noun_list = wg.get_word(5)
        prompt_lists.append(prompt_list)
        noun_lists.append(noun_list)
        
    print(prompt_lists)
    
    file_path = os.path.join(folder_name, 'prompts_test.txt')
    with open(file_path, 'w') as file:
        json.dump(prompt_lists, file)
        
    file_path = os.path.join(folder_name, 'nouns_test.txt')
    with open(file_path, 'w') as file:
        json.dump(noun_lists, file)
        
    # with open(file_path, 'r') as file:
        # loaded_word_list = json.load(file)