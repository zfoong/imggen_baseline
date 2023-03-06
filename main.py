# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 18:58:30 2023

@author: zfoong
"""

import numpy as np
from wordGen import *
from IImgGenModel import *
from DallE import *
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def der_sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))**2

# Representation Inclusive Score
def RIS(prob_vector, smoothness=1):
    weights = half_ellipse(prob_vector)
    score = np.mean(weights)
    return score

# def semi_circle_function(x):
#     l = (1/len(x))
#     scale_factor = 1 / (1 + 100 * np.abs(x - l))  # scale_factor ranges from 0 to 1
#     return scale_factor * np.sqrt(1.0 - (x - l) ** 2)

def half_ellipse(x):
    r = 1
    a = 0.5
    b = 4
    y = np.sqrt(-b*np.square(x-a)+r)
    return y

# \sqrt{-4(x-0.5)^{2}+1} = y

# def semi_circle_function(x):
#     # l = (1/len(x))
#     l = 0.5
#     scale_factor = 1 / (1 * np.abs((x - l)/l))
#     return scale_factor * np.sqrt(1.0 - (x - l) ** 2)

# initiate global param
word_gen_counts = 10 # N
gen_count = 10 # M
word_counts = 5 # K
starting_seed = 0

# initiate result collector
result_list = []

# initialize models
model = DallE()

for i in range(word_gen_counts):
    # generate random word
    wg = wordGen('simple', starting_seed)
    # caption = wg.get_word(word_counts)
    caption = ['a','a','a','a','a']
    
    for j in range(gen_count):
        # models inference
        out = model.generate(str(caption))
        
        # get word prob with CLIP
        # prob_vec = np.array([0.1,0.2,0.3,0.4,0.1])#clip.input(out)
        # prob_vec = np.array([0,0.0,0.0,0.0,1])#clip.input(out)
        prob_vec = np.array([0.2,0.2,0.2,0.2,0.2])#clip.input(out)
        
        # evaluate 
        s = RIS(prob_vec)
        print(s)
        result_list.append(s)
        
    starting_seed += 1
        
model_score = np.mean(result_list)
print("The score is " + str(model_score))
