# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:17:01 2023

@author: zfoong
"""
from min_dalle import MinDalle
import torch
import matplotlib.pyplot as plt
from IImgGenModel import *
 
class DallE_mini(IImgGenModel):

    def __init__(self):
        
        self.name = 'dalle_mini'
        # Model references
        self.model = None
        
    def init_model(self):
        self.model = MinDalle(
            models_root='./pretrained',
            dtype=torch.float32,
            device='cuda',
            is_mega=True, 
            is_reusable=True
        )

    def generate(self, caption):
        image = self.model.generate_image(
        text=caption,
        seed=-1,
        grid_size=1,
        is_seamless=False,
        temperature=1,
        top_k=128, # 256
        supercondition_factor=32,
        is_verbose=False
        )
        return image
        

if __name__ == "__main__":
    model = DallE_mini()
    image = model.generate("a man sitting on a chair")
    # plt.imshow(image)
