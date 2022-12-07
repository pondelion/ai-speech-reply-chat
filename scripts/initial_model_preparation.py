import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tacotron2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tacotron2', 'waveglow'))
import re
import numpy as np
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

import pyopenjtalk


# print(f'loading waveglow model')
# # waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
# # # waveglow.cuda().eval().half()
# # waveglow.eval().half()
# waveglow = torch.hub.load(
#     "NVIDIA/DeepLearningExamples:torchhub",
#     "nvidia_waveglow",
#     model_math="fp32",
#     pretrained=False,
# )
# checkpoint = torch.hub.load_state_dict_from_url(
#     "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_fp32/versions/19.09.0/files/nvidia_waveglowpyt_fp32_20190427",
#     progress=False,
#     map_location='cpu',
# )
# state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

# waveglow = waveglow.remove_weightnorm(waveglow)
# waveglow = waveglow.to(device)
# waveglow.eval()
# for k in waveglow.convinv:
#     k.float()
# denoiser = Denoiser(waveglow)
# print('done')


tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-medium')
model = AutoModelForCausalLM.from_pretrained('rinna/japanese-gpt2-medium')
