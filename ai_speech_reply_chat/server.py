import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.responses import Response

from ai_agent import (
    AIAgent,
    get_ai_agent,
    # create_jsut_jp_agent,
    # create_tsukiyomi_jp_agent,
    # create_default_en_agent,
    # create_jsut_tsukiyomi_jp_agent,
)
from audio_util import signal2wav_bytes


app = FastAPI()
print('initialzing ai agent')
ai_agent_jsut = get_ai_agent(chat_model_name='akane', tts_model_name='jsut')
ai_agent_tsukiyomi = get_ai_agent(chat_model_name='akane', tts_model_name='tsukuyomi')
print('done ai agent initialization')


@app.get('/')
def health():
    return 'ok'


@app.post("/talk_to_ai", response_class=Response)
def talk_to_ai(chat_text: str):
    speech_audio, reply_text = ai_agent_tsukiyomi.talk_to_ai(chat_text)
    wav_bytes = signal2wav_bytes(speech_audio)
    print(f'{chat_text} => {reply_text}')
    return Response(content=wav_bytes, media_type='audio/wav')


@app.post("/tts", response_class=Response)
def tts(text: str):
    print(f'tts text : {text}')
    speech_audio = ai_agent_jsut.tts(text)
    wav_bytes = signal2wav_bytes(speech_audio*10)
    return Response(content=wav_bytes, media_type='audio/wav')
