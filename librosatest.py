import numpy as np
import librosa
import matplotlib.pyplot as plt

file = "train\TRAIN_0000.wav" 
sg, sr = librosa.load(file, sr=22050)

FIG_SIZE = (10, 4)
plt.figure(figsize=FIG_SIZE)
librosa.display.waveshow(y=sg,sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show()