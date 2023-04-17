# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:17:01 2023

@author: zfoong
"""
import torch
import matplotlib.pyplot as plt
from IImgGenModel import *
import sys
import os

# dir_path = "./Models/attnGAN/"
# sys.path.append(dir_path)
 
class StableDiff(IImgGenModel):

    def __init__(self):
        
        self.name = 'StableDiff'
        # Model references
        self.model = None
        
    def init_model(self, GPU_ID=0, worker=1):
        self.model = None # initialize your model here

    def generate(self, caption):
        image = self.model.inference(caption) # Example of inferencing your image
        return image
        
if __name__ == "__main__":
    model = StableDiff()
    model.init_model()
    image = model.generate("a man sitting on a chair")
    # plt.imshow(image)
