import librosa
import numpy as np

from src.extract_features import extract_features


def preprocess_audio(audio_path, sr_target=44100):
    audio, sr = librosa.load(audio_path, sr=None)  # load audio with original sr
    if sr != sr_target:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_target)  # resample
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    # Trim silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    return audio_trimmed, sr_target
    



