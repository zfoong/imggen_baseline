# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 18:58:30 2023

@author: zfoong
"""

import numpy as np
from IImgGenModel import *
# from DallE import *
from DallE_mini import *
# from attnGAN import *
from StableDiff import *
import math
import os
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


def img_gen(gen_count, models, image_path, word_list_path, GPU_ID=0, worker=1, si=0, pc=000):
    
    # read words from S
    prompts = None
    with open(word_list_path, 'r') as file:
        prompts = json.load(file)
    
    if si != 0 and pc != 0:
        prompts = prompts[si:si+pc]
        
    print("Generate images based on captions")
    
    # Generate images based on captions
    for id_, model in enumerate(models):
        
        # clear cache
        torch.cuda.empty_cache()
        m = model()
        m.init_model(GPU_ID, worker)
        
        print("Performing inferencing for " + str(m.name) + "...")
        
        for word_list_id, word_list in enumerate(tqdm(prompts)):
            
            for j in range(gen_count):
                
                img_folder_path = os.path.join(image_path, m.name, str(word_list_id+si))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_count', type=int, default=8, help='M, Numbers of image output for each word list')
    parser.add_argument('--word_count', type=int, default=6, help='K, Length of each word list, [1,2,4,8]')
    parser.add_argument('--GPU_ID', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--model', type=str, default='StableDiff', help='Model to use (DallE_mini, StableDiff or attnGAN)')
    parser.add_argument('--worker', type=int, default=1, help='number of worker')
    parser.add_argument('--start_index', type=int, default=0, help='start index of the prompts list')
    parser.add_argument('--processing_count', type=int, default=10000, help='number of prompts to process in the prompts list')

    # Parse command-line arguments
    args = parser.parse_args()

    # Assign arguments to variables
    gen_count = args.gen_count
    word_count = args.word_count
    GPU_ID = args.GPU_ID
    model = args.model
    worker = args.worker
    start_index = args.start_index
    processing_count = args.processing_count

    models = []
    # models = [DallE_mini, attnGAN]
    
    if model == "DallE_mini":
        models.append(DallE_mini)
    elif model == "attnGAN":
        models.append(attnGAN)
    elif model == "StableDiff":
        models.append(StableDiff)
    else:
        raise Exception("model name not found") 
    
    # Image path
    image_path = "image_output"
    prompt_list_path = f"word_list/prompts_{word_count}.txt"

    img_gen(gen_count, models, image_path, prompt_list_path, GPU_ID, worker, start_index, processing_count)