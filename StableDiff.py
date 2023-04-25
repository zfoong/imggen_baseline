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
from Models.stablediffusion.scripts.interface import parse_args,load_model_from_config,put_watermark
import argparse
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import torchvision.transforms as transforms
# dir_path = "./Models/attnGAN/"
# sys.path.append(dir_path)
 
class StableDiff(IImgGenModel):

    def __init__(self):
        
        self.name = 'StableDiff'
        # Model references
        self.model = None
        self.opt = parse_args()
        
        opt = self.opt.parse_args()
        seed_everything(opt.seed)

        self.config = OmegaConf.load(f"{opt.config}")
        device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
        self.model = load_model_from_config(self.config, f"{opt.ckpt}", device)

        if opt.plms:
            self.sampler = PLMSSampler(self.model, device=device)
        elif opt.dpm:
            self.sampler = DPMSolverSampler(self.model, device=device)
        else:
            self.sampler = DDIMSampler(self.model, device=device)

        os.makedirs(opt.outdir, exist_ok=True)
        self.outpath = opt.outdir
        
        self.wm = "SDV2"
        self.wm_encoder = WatermarkEncoder()
        self.wm_encoder.set_watermark('bytes', self.wm.encode('utf-8'))
        
        self.batch_size = opt.n_samples
        self.n_rows = opt.n_rows if opt.n_rows > 0 else self.batch_size
        
        
        
        self.sample_path = os.path.join(self.outpath, "samples")
        os.makedirs(self.sample_path, exist_ok=True)
        self.sample_count = 0
        self.base_count = len(os.listdir(self.sample_path))
        self.grid_count = len(os.listdir(self.outpath)) - 1

        self.start_code = None
        if opt.fixed_code:
            self.start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        if opt.torchscript or opt.ipex:
            transformer = self.model.cond_stage_model.model
            unet = self.model.model.diffusion_model
            decoder = self.model.first_stage_model.decoder
            
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

            if opt.bf16 and not opt.torchscript and not opt.ipex:
                raise ValueError('Bfloat16 is supported only for torchscript+ipex')
            if opt.bf16 and unet.dtype != torch.bfloat16:
                raise ValueError("Use configs/stable-diffusion/intel/ configs with bf16 enabled if " +
                                 "you'd like to use bfloat16 with CPU.")
            if unet.dtype == torch.float16 and device == torch.device("cpu"):
                raise ValueError("Use configs/stable-diffusion/intel/ configs for your model if you'd like to run it on CPU.")

            if opt.ipex:
                import intel_extension_for_pytorch as ipex
                bf16_dtype = torch.bfloat16 if opt.bf16 else None
                transformer = transformer.to(memory_format=torch.channels_last)
                transformer = ipex.optimize(transformer, level="O1", inplace=True)

                unet = unet.to(memory_format=torch.channels_last)
                unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

                decoder = decoder.to(memory_format=torch.channels_last)
                decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

            if opt.torchscript:
                with torch.no_grad(), self.additional_context:
                    # get UNET scripted
                    if unet.use_checkpoint:
                        raise ValueError("Gradient checkpoint won't work with tracing. " +
                        "Use configs/stable-diffusion/intel/ configs for your model or disable checkpoint in your config.")

                    img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32)
                    t_in = torch.ones(2, dtype=torch.int64)
                    context = torch.ones(2, 77, 1024, dtype=torch.float32)
                    scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
                    scripted_unet = torch.jit.optimize_for_inference(scripted_unet)
                    # print(type(scripted_unet))
                    self.model.model.scripted_diffusion_model = scripted_unet

                    # get Decoder for first stage model scripted
                    samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
                    scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
                    scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)
                    # print(type(scripted_decoder))
                    self.model.first_stage_model.decoder = scripted_decoder
        
    def init_model(self, GPU_ID=0, worker=1):
        None
        #model = None # initialize your model here

    def generate(self, caption):
        opt = self.opt.parse_args(['--prompt',caption])
        #image = model.inferece(caption) # Example of inferencing your image
        if not opt.from_file:
            prompt = opt.prompt
            assert prompt is not None
            data = [self.batch_size * [prompt]]

        else:
            # print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = [p for p in data for i in range(opt.repeat)]
                data = list(chunk(data, self.batch_size))
        prompts = data[0]
        # print("Running a forward pass to initialize optimizations")
        uc = None
        if opt.scale != 1.0:
            uc = self.model.get_learned_conditioning(self.batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)
        # print(prompts)
        uc = None
        if opt.scale != 1.0:
            uc = self.model.get_learned_conditioning(self.batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)

        precision_scope = autocast if opt.precision=="autocast" or opt.bf16 else nullcontext
        
        with torch.no_grad(), \
            precision_scope(opt.device), \
            self.model.ema_scope():
                all_samples = list()
                for n in range(opt.n_iter):
                    for prompts in data:
                        uc = None
                        if opt.scale != 1.0:
                            uc = self.model.get_learned_conditioning(self.batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples, _ = self.sampler.sample(S=opt.steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=self.start_code)

                        x_samples = self.model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            self.base_count += 1
                            self.sample_count += 1

                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=self.n_rows)
                
                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                #print("where am i ",grid.shape)
                #to_pil = transforms.ToPILImage()
                #img = to_pil(grid)
                self.grid_count += 1
        return grid 

        
if __name__ == "__main__":
    model = StableDiff()
    img = model.generate("a professional photograph of an astronaut riding a horse")
