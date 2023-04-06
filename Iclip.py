#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:41:50 2023

@author: bob
"""

import torch
import clip
from PIL import Image
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime 



def add_article(word):
    vowels = ['a', 'e', 'i', 'o', 'u']
    if word[0].lower() in vowels:
        return 'an ' + word
    else:
        return 'a ' + word


class Iclip():
    
    def __init__(self, show_chart=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.show_chart = show_chart
        
    
    def input(self, I_path, T):
        image = self.preprocess(Image.open(I_path)).unsqueeze(0).to(self.device)
        # word_list = T.copy()
        # word_list.append("")
        
        combination_list = []
        for i in range(1, len(T) + 1):
            combination_list += list(itertools.combinations(T, i))
        
        formatted_word_list = [list(x) for x in combination_list]
        
        word_list = []
        noun_counts = []
        
        for w_list in formatted_word_list:
            
            noun_counts.append(len(w_list))
            
            formatted_nouns = [add_article(n) for n in w_list]
            
            template = "A photo of {}."
            prompt = None
            if len(formatted_nouns) > 1:
                last_item = formatted_nouns.pop()
                prompt = template.format(', '.join(formatted_nouns) + ', and ' + last_item)
            else:
                prompt = template.format(formatted_nouns[0])
            word_list.append(prompt)
        
        # Adding "" into the prompt account for image without any nouns in it.
        word_list.append("")
        noun_counts.append(0)
        
        text = clip.tokenize(word_list).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
        pid = np.argmax(probs[0])
        noun_count = noun_counts[pid]
        # s = noun_count/max(noun_counts)
        
        confidence = np.max(probs)
            
        # print("T:", T)
        # print("Label probs:", probs)

        if self.show_chart:
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16), dpi=300)
            
            # Plot the image on top
            img = mpimg.imread(I_path)
            ax1.imshow(img)
            ax1.axis('off')
            
            # Plot the horizontal bar chart at the bottom
            ax2.barh(word_list, probs[0])
            ax2.set_xlabel('Probs')
            ax2.set_ylabel('Prompts')

            plt.subplots_adjust(hspace=0.3)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig('prob_output/img_{}.png'.format(timestamp), bbox_inches='tight')
            plt.show()
        
        return noun_count, confidence


