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
from attnGAN import *
import math
import os
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

from Iclip import *

def RIS(noun_count, K):
    score = noun_count/K
    return score

def img_eval(gen_count, word_counts, models, image_path, word_list_path):
    
    # initiate result collector
    result_list = []
    confidence_list = []
    
    print("Evaluating RIS")
    
    # read nouns from 
    word_data = None
    with open(word_list_path, 'r') as file:
        word_data = json.load(file)
    
    torch.cuda.empty_cache()
    clip = Iclip()
    # Compute RIS
    for id_, model in enumerate(models):
    
        m = model()
        print("Evaluating " + str(m.name) + "...")
        
        for word_list_id, word_list in enumerate(tqdm(word_data)):
            
            # print("word list: " + str(word_list))
            
            for j in range(gen_count):
                # models inference
                img_path = os.path.join(image_path, m.name, str(word_list_id), "img_"+str(j)+".jpg") 
                # img_out = Image.open(img_path) 
                
                # get nouns count present in the image with CLIP
                noun_count, confidence = clip.input(img_path, word_list) 
                
                # print("noun count: " + str(noun_count))
    
                # evaluate 
                s = RIS(noun_count, word_counts)
                result_list.append(s)
                
                confidence_list.append(confidence)
                
        score_mean = np.mean(result_list)
        score_std = np.std(result_list)
        
        confidence_mean = np.mean(confidence_list)
        confidence_std = np.std(confidence_list)
    
        print(f'The score for {m.name} is {score_mean:.3f}±{score_std:.3f}')
        print(f'The confidence for {m.name} is {confidence_mean:.3f}±{confidence_std:.3f}')
        
        print("Evaluating " + str(m.name) + " completed")
    
    print("Evaluating RIS completed")

if __name__ == "__main__":
    
    # initiate global param
    gen_count = 3 # M, Numbers of image output for each word list
    word_counts = 5 # K, Length of each word list

    models = [DallE_mini, attnGAN]
    
    # Image path
    image_path = "image_output"
    noun_list_path = "word_list/nouns_test.txt"

    img_eval(gen_count, word_counts, models, image_path, noun_list_path)