# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 18:58:30 2023

@author: zfoong
"""

import numpy as np
from wordGen import *
from IImgGenModel import *
# from DallE import *
from DallE_mini import *
import math
import os
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

from Iclip import *

# Representation Inclusive Score
def RIS(prob_vector, smoothness=1):
    weights = half_ellipse(prob_vector)
    score = np.mean(weights)
    return score

def half_ellipse(x):
    r = 1
    a = 1/len(x)
    b = np.square(len(x))
    y = -b*np.square(x-a)+r
    y[np.where(y < 0)] = 0
    return np.sqrt(y)

# initiate global param
# word gen counts is 10 in words_test.txt and 10000 in words.txt
# word_gen_counts = 1 # N
gen_count = 10 # M, Numbers of image output for each word list
word_counts = 5 # K, Length of each word list

# initiate result collector
result_list = []

# model list
models = [DallE_mini]

# Image path
image_path = "image_output"

# read words from 
word_data = None
word_list_path = "word_list/words_test.txt"
with open(word_list_path, 'r') as file:
    word_data = json.load(file)

print("Generate images based on captions")

# Generate images based on captions
for id_, model in enumerate(models):
    
    # clear cache
    # torch.cuda.empty_cache()
    m = model()
    m.init_model()
    
    print("Performing inferencing for " + str(m.name) + "...")
    
    for word_list_id, word_list in enumerate(tqdm(word_data)):
        
        for j in range(gen_count):
            
            img_folder_path = os.path.join(image_path, m.name, str(word_list_id))
            if not os.path.exists(img_folder_path):
                os.makedirs(img_folder_path)
            
            img_path = os.path.join(img_folder_path, "img_"+str(j)+".jpg")
            # models inference
            img_out = m.generate(' '.join(word_list))
            
            # In case image output is not a PIL object, convert the tensor to a PIL image
            # to_pil = transforms.ToPILImage()
            # img = to_pil(img_out)
            
            # Save the PIL image to a file
            img_out.save(img_path, 'JPEG')
        
    print("Inferencing for " + str(m.name) + " Completed")
    
print("Images generation based on captions completed")


print("Evaluating RIS")

torch.cuda.empty_cache()
clip = Iclip()
# Compute RIS
for id_, model in enumerate(models):

    m = model()
    print("Evaluating " + str(m.name) + "...")
    
    for word_list_id, word_list in enumerate(tqdm(word_data)):
        
        for j in range(gen_count):
            # models inference
            img_path = os.path.join(image_path, m.name, str(word_list_id), "img_"+str(j)+".jpg") 
            # img_out = Image.open(img_path) 
            
            # get word prob with CLIP
            prob_vec = clip.input(img_path, word_list) 
            
            # evaluate 
            s = RIS(prob_vec)
            result_list.append(s)
            
    score_mean = np.mean(result_list)
    score_std = np.std(result_list)
    print(f'The score for {m.name} is {score_mean:.3f}±{score_std:.3f}')
    
    print("Evaluating " + str(m.name) + " completed")

print("Evaluating RIS completed")
