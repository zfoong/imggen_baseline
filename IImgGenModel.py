# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:15:32 2023

@author: zfoong
"""

from abc import ABC, abstractmethod

class IImgGenModel(ABC):
    
    @abstractmethod
    def __init__(self):
        self.name = None
        pass    
    
    @abstractmethod
    def generate(self, caption):
        pass