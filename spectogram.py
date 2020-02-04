import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from librosa.feature import zero_crossing_rate, melspectrogram
import librosa
from sklearn.decomposition import FastICA

sr = 16000
n_data=500
n_mel=128
n_fft=2048
win_length=2048
hop_length=1024
slice_dur=3.3
freq_range=(0, 8000)
chunk_dur = 1

audio_params = {
    "sr": sr,
    "win_length": win_length,
    "hop_length": hop_length,
    "n_mel": n_mel,
    "n_fft": n_fft,
    "f_min": freq_range[0],
    "f_max": freq_range[1]
}

config = {
    "n_input": n_mel,
    "n_output": None,
    "n_hidden": 256,
    "n_layer": 5,
    "dropout": 0.2,
    "audio_params": audio_params,
    "classes": None
}

wav_file='ravdess-emotional-speech-audio/Actor_01/03-01-01-01-01-01-01.wav'

sr = audio_params["sr"]
win_length = audio_params["win_length"]
hop_length = audio_params["hop_length"]
n_mel = audio_params["n_mel"]
n_fft = audio_params["n_fft"]
f_min = audio_params["f_min"]
f_max = audio_params["f_max"]
chunk_len = chunk_dur * sr

def preprocess(wav_path, sr):
    _X, sr = librosa.load(wav_path, sr)
    X = quantile_normalize(_X)
    X = padding(X, chunk_len)
    return _X, X, sr

def padding(audio, chunk_len):
    mod = (len(audio) % chunk_len)
    if mod != 0:
        pad = np.ones(chunk_len - mod)*1e-5
        audio = np.concatenate([audio, pad])
    return audio.reshape(-1, chunk_len)

def spectogram(audio):
    spec = melspectrogram(audio, sr=audio_params['sr'], center=True, 
          n_mels=audio_params['n_mel'], n_fft=audio_params['n_fft'], 
          win_length=audio_params['win_length'], hop_length=audio_params['hop_length'], 
          fmin=audio_params['f_min'], 
          fmax=audio_params['f_max'])
    spec = np.log(spec)

def quantile_normalize(audio, quantile=0.999):
    return audio / np.quantile(abs(audio), quantile)

def zcr(audio, shift=0.05, frame_len=2048, hop_len=1024, zcr_threshold=0.005):
    zc_rate = zero_crossing_rate(audio+shift, frame_length=frame_len, hop_length=hop_len, center=False)[0]
    mask = np.where(zc_rate > zcr_threshold, 1, 0)
    return mask