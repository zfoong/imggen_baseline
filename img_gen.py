# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 18:58:30 2023

@author: zfoong
"""

import numpy as np
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

def img_gen(gen_count, word_counts, models, image_path, word_list_path):
    
    # read words from S
    prompts = None
    with open(word_list_path, 'r') as file:
        prompts = json.load(file)
    
    print("Generate images based on captions")
    
    # Generate images based on captions
    for id_, model in enumerate(models):
        
        # clear cache
        torch.cuda.empty_cache()
        m = model()
        m.init_model()
        
        print("Performing inferencing for " + str(m.name) + "...")
        
        for word_list_id, word_list in enumerate(tqdm(prompts)):
            
            for j in range(gen_count):
                
                img_folder_path = os.path.join(image_path, m.name, str(word_list_id))
                if not os.path.exists(img_folder_path):
                    os.makedirs(img_folder_path)
                
                img_path = os.path.join(img_folder_path, "img_"+str(j)+".jpg")
                # models inference
                img_out = m.generate(word_list)
                
                # In case image output is not a PIL object, convert the tensor to a PIL image
                # to_pil = transforms.ToPILImage()
                # img = to_pil(img_out)
                
                # Save the PIL image to a file
                img_out.save(img_path, 'JPEG')
            
        print("Inferencing for " + str(m.name) + " Completed")
        
    print("Images generation based on captions completed")

if __name__ == "__main__":
    
    # initiate global param
    gen_count = 3 # M, Numbers of image output for each word list
    word_counts = 5 # K, Length of each word list

    models = [DallE_mini, attnGAN]
    
    # Image path
    image_path = "image_output"
    prompt_list_path = "word_list/prompts_test.txt"

    img_gen(gen_count, word_counts, models, image_path, prompt_list_path)