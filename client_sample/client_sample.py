import io

import requests
import sounddevice as sd
import soundfile as sf


res = requests.post('http://localhost:8000/tts?text=こんにちは。私の名前は田中です。')
print(res.content[:10])

data, fs = sf.read(file=io.BytesIO(res.content), dtype='float64')  
sd.play(data, fs)
status = sd.wait() 


res = requests.post('http://localhost:8000/talk_to_ai?chat_text=ご機嫌いかが？')
print(res.content[:10])

data, fs = sf.read(file=io.BytesIO(res.content), dtype='float64')
print(fs)
sd.play(data, fs)
status = sd.wait() 
