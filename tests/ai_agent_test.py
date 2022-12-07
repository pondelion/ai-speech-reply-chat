import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_speech_reply_chat'))

from ai_agent import AIAgent

ai_agent = AIAgent()
speech_audio, reply_text = ai_agent.talk_to_ai('こんにちは。')
print(reply_text)
print(speech_audio)
import soundfile as sf
import numpy as np
from io import BytesIO
sf.write(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'test.wav'), speech_audio*10, 22050, subtype="PCM_16")

buffer = BytesIO()
sf.write(buffer, speech_audio*10, 22050, subtype="PCM_16", format='wav')
buffer.seek(0)
print(buffer.read()[:10])