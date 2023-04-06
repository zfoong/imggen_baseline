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

from img_eval import *
from img_gen import *

def main():
    generate = False
    evaluate = True
    
    # initiate global param
    # word gen counts is 10 in prompts_test.txt and 10000 in prompts.txt
    # word_gen_counts = 1 # N
    gen_count = 100 # M, Numbers of image output for each word list
    word_counts = 5 # K, Length of each word list
    
    # model list
    # models = [attnGAN]
    models = [DallE_mini, attnGAN]
    
    # Image path
    image_path = "image_output"
    prompt_list_path = "word_list/prompts_test.txt"
    noun_list_path = "word_list/nouns_test.txt"
    
    if generate:
        img_gen(gen_count, word_counts, models, image_path, prompt_list_path)
        
    if evaluate:
        img_eval(gen_count, word_counts, models, image_path, noun_list_path)

if __name__ == "__main__":
    main()


