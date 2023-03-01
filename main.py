# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 18:58:30 2023

@author: zfoong
"""

import numpy as np
from wordGen import *
from IImgGenModel import *
from DallE import *

def diversity_score(prob_vec):
    return 0


# initiate global param
word_gen_counts = 10
gen_count = 10
word_counts = 5
starting_seed = 0

# initiate result collector
ds_list = []

# initialize models
model = DallE()

for i in range(word_gen_counts):
    # generate random word
    wg = wordGen('simple', starting_seed)
    caption = wg.get_word(word_counts)
    
    for j in range(gen_count):
        # models inference
        out = model.generate(str(caption))
        
        # get word prob with CLIP
        prob_vec = 0#clip.input(out)
        
        # evaluate 
        s = diversity_score(prob_vec)
        ds_list.append(s)
        
    starting_seed += 1
        
ds = np.mean(ds_list)
print("The score is " + str(ds))
