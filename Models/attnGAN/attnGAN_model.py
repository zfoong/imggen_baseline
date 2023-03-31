from __future__ import print_function

# From libraries
import argparse
import datetime
import os
import pprint
import random
import sys
import time

import dateutil.tz
import numpy as np

import torch
import torchvision.transforms as transforms
from nltk.tokenize import RegexpTokenizer

# From files
from datasets import TextDataset
from miscc.config import cfg, cfg_from_file
from trainer import condGANTrainer as trainer

def gen_example_from_text(input_text, output_dir, wordtoix, algo):
    '''
    Generate image from example sentence
    '''

    # a list of indices for a sentence
    captions = []
    cap_lens = []
    data_dic = {}
    if len(input_text) == 0:
        return 0
    sent = input_text.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sent.lower())
    if len(tokens) == 0:
        print('No tokens for: ', sent)
        return 0

    rev = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            rev.append(wordtoix[t])
    captions.append(rev)
    cap_lens.append(len(rev))
    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    data_dic["data"] = [cap_array, cap_lens, sorted_indices]
    return algo.gen_example(output_dir, data_dic)

class attnGAN_model():
    
    def __init__(self):
        # dir_path = (os.path.abspath(os.path.join(os.path.dirname(__file__), './.')))
        # sys.path.append(dir_path)
        
        cfg_file = 'cfg/eval_coco.yml'
        cfg_file = os.path.join(os.path.dirname(__file__), cfg_file)

        cfg_from_file(cfg_file)
    
        cfg.GPU_ID = 0
        cfg.DATA_DIR = 'data/coco'
        cfg.DATA_DIR = os.path.join(os.path.dirname(__file__), cfg.DATA_DIR)
        
        cfg.TRAIN.NET_G = 'models/coco_AttnGAN2.pth'
        cfg.TRAIN.NET_G = os.path.join(os.path.dirname(__file__), cfg.TRAIN.NET_G)
        
        cfg.TRAIN.NET_E = 'DAMSMencoders/coco/text_encoder100.pth'
        cfg.TRAIN.NET_E = os.path.join(os.path.dirname(__file__), cfg.TRAIN.NET_E)
    
        manualSeed = random.randint(1, 10000)
        # manualSeed = 100
        
        random.seed(manualSeed)
        np.random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        if cfg.CUDA:
            torch.cuda.manual_seed_all(manualSeed)

        self.output_dir = './output/'
    
        self.split_dir, bshuffle = 'train', True
        if not cfg.TRAIN.FLAG:
            # bshuffle = False
            self.split_dir = 'test'
            
        # Get data loader
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        self.dataset = TextDataset(cfg.DATA_DIR, self.split_dir,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        assert self.dataset
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
    
        # Define models and go to train/evaluate
        self.model = trainer(self.output_dir, dataloader, self.dataset.n_words, self.dataset.ixtoword)
    
    def inference(self, input_text):
        return gen_example_from_text(input_text, self.output_dir, self.dataset.wordtoix, self.model)
        

if __name__ == "__main__":
    model = attnGAN_model()
    img = model.inference("square pizza")
