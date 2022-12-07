from io import BytesIO

import soundfile as sf
import numpy as np


def signal2wav_bytes(audio_signal: np.ndarray) -> BytesIO:
    buffer = BytesIO()
    sf.write(buffer, audio_signal, 22050, subtype="PCM_16", format='wav')
    buffer.seek(0)
    wav_bytes = buffer.read()
    buffer.close()
    return wav_bytes


def signal2wav_file(audio_signal: np.ndarray, output_filepath) -> None:
    sf.write(output_filepath, audio_signal, 22050, subtype="PCM_16")
