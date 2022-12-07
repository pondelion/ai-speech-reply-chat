import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tacotron2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tacotron2', 'waveglow'))
import re
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

import pyopenjtalk


def preprocess_text(text):
    text = pyopenjtalk.g2p(text, kana=False)
    text = re.sub('・|・|「|」|』|(|)|（|）', '', text)
    text = text.replace('。', '、')
    text = text.replace('pau',',')
    text = text.replace(' ','')
    text = text + '.'
    return text

hparams = create_hparams()
hparams.sampling_rate = 22050

print(f'loading tacotron model')
checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tacotron2_statedict.pt')
tacotron_model = load_model(hparams)
tacotron_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = tacotron_model.cuda().eval().half()
print('done')

print(f'loading waveglow model')
# waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow', pretrained=False)
checkpoint = torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'nvidia_waveglowpyt_fp32_20190427'))
state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
waveglow.load_state_dict(
    state_dict,
    # map_location='cuda',
)
# state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)
print('done')


text = preprocess_text('こんにちは')
print('start converting text to spectrogram')
sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()
mel_outputs, mel_outputs_postnet, _, alignments = tacotron_model.inference(sequence)
print('done')

print('start converting spectrogram to audio signal')
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
print('done')

print(audio)

from scipy.io.wavfile import read, write
import soundfile as sf
import numpy as np
sf.write(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'test.wav'), audio.cpu().detach().numpy().transpose().astype(np.float32), 22050, subtype="PCM_16")