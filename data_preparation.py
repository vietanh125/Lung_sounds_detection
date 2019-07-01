import tensorflow
import librosa
from scipy import signal
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Model, Sequential
X = []
y = []
plt.figure()

c1, c2, c3, c4 = 0, 0, 0, 0
ANNO_PATH = "/home/nguyen.viet.anhd/Downloads/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/annotation/"
AU_PATH = "/home/nguyen.viet.anhd/Downloads/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio/"
for name in os.listdir(AU_PATH):
    print(len(X), len(y))
    name = name.split(".")[0]
    au, sr = librosa.load(AU_PATH + name + ".wav", 8000)
    anno = np.genfromtxt(open(ANNO_PATH + name + ".txt", "rb"))
    for row in anno:
        start, end, cr, wh = row
        frames = au[int(start*sr): int(end*sr)+1]
        if len(frames) < 2700:
            frames = np.append(frames, [0]*(2700-len(frames)))
        frames = librosa.util.frame(frames, 2700, 1600)
        frames = np.transpose(frames)
        # print(frames.shape)
        #
        for frame in frames:
            # print(np.array(frame).shape)
            if cr == 0 and wh == 0:
                # print("normal")
                y.append(0)
                c1 += 1
            elif cr == 0 and wh == 1:
                # print("wheeze")
                y.append(1)
                c2 += 1
            elif cr == 1 and wh == 0:
                # print("crackle")
                y.append(2)
                c3 += 1
            else:
                continue
                # y.append(3)
                # c4 += 1
            X.append(frame)
            # print("loading...", len(X), "samples")
            # frame = librosa.amplitude_to_db(np.abs(librosa.stft(frame)))
            # print(frame.shape)
            frame = np.log(librosa.feature.melspectrogram(frame, sr, n_fft=200, hop_length=120, n_mels=60) + 1e-10)
            # print(frame.shape)
            # print(cr, wh)
            # print(frame)
            #
            # librosa.display.specshow(frame, y_axis='linear')
            # plt.show()
X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)
np.save("sliced.npy", X)
np.save("y_sliced.npy", y)
# X_new = []
X = np.load("sliced_new.npy", allow_pickle=True)
y = np.load("y_sliced.npy", allow_pickle=True)
print(X.shape, y.shape)
# for frame in X:
#     delta = librosa.feature.delta(frame)
#     frame = np.expand_dims(frame, 2)
#     delta = np.expand_dims(delta, 2)
#     X_new.append(np.append(frame, delta, axis=2))
#     if len(X_new) % 1000 == 0:
#         print(len(X_new))
# X_new = np.array(X_new)
# print(X_new.shape, y.shape)
# np.save("sliced_new.npy", X_new)
