import subprocess
# import os.path as osp
import pip
# pip.main(["install","-v","-U","git+https://github.com/facebookresearch/xformers.git@main#egg=xformers"])
# subprocess.check_call("pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers", cwd=osp.dirname(__file__), shell=True)

import io
import base64
import os
import sys

import numpy as np
import torch
from torch import autocast
import diffusers
from diffusers.configuration_utils import FrozenDict
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    DDIMScheduler,
    LMSDiscreteScheduler,
    StableDiffusionUpscalePipeline,
    DPMSolverMultistepScheduler
)
from diffusers.models import AutoencoderKL
from PIL import Image
from PIL import ImageOps
import gradio as gr
import base64
import skimage
import skimage.measure
import yaml
import json
from enum import Enum
from convert_checkpoint import *

try:
    cuda_available = torch.cuda.is_available()
except:
    cuda_available = False
finally:
    if sys.platform == "darwin":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    elif cuda_available:
        device = "cuda"
    else:
        device = "cpu"
        
        
class StableDiffusionInpaint:
    def __init__(
        self, token: str = "",
        sd_model_name: str = "",
        model_path: str = "",
        vae_model_path: str = "",
        **kwargs,
    ):
        self.token = token
        original_checkpoint = True # 使用本地文件
        
        if not os.path.exists(model_path):
            print("model_path error, please input the correct model_path")
        else:
            sd_model_name = model_path
        
        ''' Convert the VAE model.
        if len(vae_model_path) > 0 and os.path.exists(vae_model_path)
            vae_config = create_vae_diffusers_config(original_config)
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

            vae = AutoencoderKL(**vae_config)
            vae.load_state_dict(converted_vae_checkpoint)
        '''
        
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.to(torch.float16)
        if original_checkpoint: # True: use local model
            print(f"Converting & Loading {model_path}")
            from convert_checkpoint import convert_checkpoint

            pipe = convert_checkpoint(model_path, inpainting=True)
            if device == "cuda":
                pipe.to(torch.float16)
            inpaint = StableDiffusionInpaintPipeline(
                vae=vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
            )