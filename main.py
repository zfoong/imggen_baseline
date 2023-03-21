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
word_gen_counts = 1 # N
gen_count = 1 # M
word_counts = 5 # K

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

# Generate images based on captions
# for id_, model in enumerate(models):
    
#     # clear cache
#     # torch.cuda.empty_cache()
#     m = model()
#     m.init_model()
    
#     for word_list_id, word_list in enumerate(word_data):
        
#         for j in range(gen_count):
            
#             img_folder_path = os.path.join(image_path, m.name, str(word_list_id))
#             if not os.path.exists(img_folder_path):
#                 os.makedirs(img_folder_path)
            
#             img_path = os.path.join(img_folder_path, "img_"+str(j)+".jpg")
#             # models inference
#             img_out = m.generate(' '.join(word_list))
            
#             # In case image output is not a PIL object, convert the tensor to a PIL image
#             # to_pil = transforms.ToPILImage()
#             # img = to_pil(img_out)
            
#             # Save the PIL image to a file
#             img_out.save(img_path, 'JPEG')

torch.cuda.empty_cache()
clip = Iclip()
# Compute RIS
for id_, model in enumerate(models):

    m = model()
    
    for word_list_id, word_list in enumerate(word_data):
        
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
    print(f'The score for {m.name} is {score_mean:.2f}±{score_std:.2f}')
