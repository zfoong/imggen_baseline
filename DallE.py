# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:17:01 2023

@author: zfoong
"""

from IImgGenModel import *


import sys

sys.path.insert(1, 'Models/dalle')
from inference_dalle import *
 
class DallE(IImgGenModel):

    def __init__(self):
        self.dalle = inference_dalle()

    def generate(self, caption):
        output = self.dalle.inference(caption)
        return output
    
dalle = DallE()
dalle.generate("a man sitting on a chair")