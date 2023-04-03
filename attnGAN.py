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

dir_path = "./Models/attnGAN/"
sys.path.append(dir_path)

from attnGAN_model import *
 
class attnGAN(IImgGenModel):

    def __init__(self):
        
        self.name = 'attnGAN'
        # Model references
        self.model = None
        
    def init_model(self):
        self.model = attnGAN_model()

    def generate(self, caption):
        image = self.model.inference(caption)
        return image
        
if __name__ == "__main__":
    model = attnGAN()
    model.init_model()
    image = model.generate("a man sitting on a chair")
    # plt.imshow(image)
