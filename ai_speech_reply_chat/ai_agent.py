import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tacotron2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tacotron2', 'waveglow'))
import re
import numpy as np
import torch
import pyopenjtalk
from transformers import T5Tokenizer, AutoModelForCausalLM
import librosa

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

from logger_util import Logger


class ChatbotLanguageModel:

    def __init__(
        self,
        # lm_gpt2_model_fileptah: str = os.path.join(os.path.dirname(__file__), '..', 'models', 'rinna_gpt2_medium_model.pth'),
        lm_gpt2_model_fileptah: str = os.path.join(os.path.dirname(__file__), '..', 'models', 'rinna_gpt2_akane_finetuned'),
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self._device = device
        if lm_gpt2_model_fileptah.endswith('.pth') or lm_gpt2_model_fileptah.endswith('.pt'):
            self._lm_gpt2_model = AutoModelForCausalLM.from_pretrained('rinna/japanese-gpt2-medium')
            self._lm_gpt2_model.load_state_dict(torch.load(lm_gpt2_model_fileptah))
        else:
            print(f'loading {lm_gpt2_model_fileptah}')
            self._lm_gpt2_model = AutoModelForCausalLM.from_pretrained(lm_gpt2_model_fileptah)
        _ = self._lm_gpt2_model.eval().to(device)
        self._tokenizer = T5Tokenizer.from_pretrained('rinna/japanese-gpt2-medium')

    def chat(self, text: str) -> str:
        input = self._tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            output = self._lm_gpt2_model.generate(input.to(self._device), do_sample=True, max_length=100, num_return_sequences=1)
            reply_text = self._tokenizer.batch_decode(output)[0]
        reply_text = reply_text.split('</s>')[1]
        return reply_text


class TTSModel:

    def __init__(
        self,
        tacotron_model_filepath: str = os.path.join(os.path.dirname(__file__), '..', 'models', 'tacotron2_checkpoint_4000_jsut.pth'),
        # tacotron_model_filepath: str = os.path.join(os.path.dirname(__file__), '..', 'models', 'tacotron2_checkpoint_1000_tsukuyomi.pth'),
        waveglow_model_fileptah: str = os.path.join(os.path.dirname(__file__), '..', 'models', 'nvidia_waveglowpyt_fp32_20190427'),
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        audio_scale: float = 0.8,
    ):
        self._device = device
        self._device = device
        self._hparams = create_hparams()
        self._hparams.sampling_rate = 22050
        self._tacotron_model = load_model(self._hparams)
        self._tacotron_model.load_state_dict(torch.load(tacotron_model_filepath)['state_dict'])
        _ = self._tacotron_model.to(device).eval().half()
        self._waveglow_model = self._load_waveglow(waveglow_model_fileptah, device)
        self._denoiser = Denoiser(self._waveglow_model)
        self._audio_scale = audio_scale

    def tts(self, text: str) -> np.ndarray:
        text_processed = self._preprocess_text(text)

        sequence = np.array(text_to_sequence(text_processed, ['basic_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(self._device).long()
        mel_outputs, mel_outputs_postnet, _, alignments = self._tacotron_model.inference(sequence)

        with torch.no_grad():
            audio = self._waveglow_model.infer(mel_outputs_postnet, sigma=0.666)
        audio = audio[0].data.cpu().numpy().transpose().astype(np.float32)
        audio = self._audio_scale * librosa.util.normalize(audio)
        return audio

    def _preprocess_text(self, text: str) -> str:
        sentences = ''
        for t in text.split('。'):
            if len(sentences) + len(t) > 70:
                break
            sentences += t + '。'
        text = sentences
        text = pyopenjtalk.g2p(text, kana=False)
        text = re.sub('・|・|「|」|』|(|)|（|）', '', text)
        text = text.replace('。', '、')
        text = text.replace('pau',',')
        text = text.replace(' ','')
        text = text + '.'
        return text

    def _load_waveglow(self, model_filepath, device):
        waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow', pretrained=False)
        checkpoint = torch.load(model_filepath, map_location=device)
        state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
        waveglow.load_state_dict(state_dict)
        waveglow.to(device).eval().half()
        for k in waveglow.convinv:
            k.float()
        return waveglow


class AIAgent:

    def __init__(
        self,
        chat_model: ChatbotLanguageModel,
        tts_model: TTSModel
    ):
        self._chat_model = chat_model
        self._tts_model = tts_model

    def talk_to_ai(self, text):
        Logger.d('AIAgent.talk_to_ai', 'start inference')
        reply_text = self._chat_model.chat(text)
        audio = self._tts_model.tts(reply_text)
        Logger.d('AIAgent.talk_to_ai', 'done inference')
        return audio, reply_text

    def tts(self, text: str) -> np.ndarray:
        return self._tts_model.tts(text)

    def chat(self, text: str) -> str:
        return self._chat_model.chat(text)


chat_models = {
    'rinna': None,
    'akane': None,
}
tts_models = {
    'en': None,
    'jsut': None,
    'tsukuyomi': None,
}


def get_ai_agent(chat_model_name: str, tts_model_name: str) -> AIAgent:
    if chat_models[chat_model_name] is None:
        if chat_model_name == 'rinna':
            chat_models[chat_model_name] = ChatbotLanguageModel(
                lm_gpt2_model_fileptah=os.path.join(os.path.dirname(__file__), '..', 'models', 'rinna_gpt2_medium_model.pth'),
            )
        elif chat_model_name == 'akane':
            chat_models[chat_model_name] = ChatbotLanguageModel(
                lm_gpt2_model_fileptah=os.path.join(os.path.dirname(__file__), '..', 'models', 'rinna_gpt2_akane_finetuned'),
            )
        else:
            raise Exception('unexpected')
    if tts_models[tts_model_name] is None:
        if tts_model_name == 'en':
            tts_models[tts_model_name] = TTSModel(
                tacotron_model_filepath=os.path.join(os.path.dirname(__file__), '..', 'models', 'tacotron2_statedict.pt'),
                waveglow_model_fileptah=os.path.join(os.path.dirname(__file__), '..', 'models', 'nvidia_waveglowpyt_fp32_20190427'),
            )
        elif tts_model_name == 'jsut':
            tts_models[tts_model_name] = TTSModel(
                tacotron_model_filepath=os.path.join(os.path.dirname(__file__), '..', 'models', 'tacotron2_checkpoint_4000_jsut.pth'),
                waveglow_model_fileptah=os.path.join(os.path.dirname(__file__), '..', 'models', 'nvidia_waveglowpyt_fp32_20190427'),
            )
        elif tts_model_name == 'tsukuyomi':
            tts_models[tts_model_name] = TTSModel(
                tacotron_model_filepath=os.path.join(os.path.dirname(__file__), '..', 'models', 'tacotron2_checkpoint_1000_tsukuyomi.pth'),
                waveglow_model_fileptah=os.path.join(os.path.dirname(__file__), '..', 'models', 'nvidia_waveglowpyt_fp32_20190427'),
            )
        else:
            raise Exception('unexpected')
    return AIAgent(
        chat_model=chat_models[chat_model_name], tts_model=tts_models[tts_model_name],
    )
