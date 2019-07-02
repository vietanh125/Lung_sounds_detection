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
c1, c2, c3, c4 = 0, 0, 0, 0
ANNO_PATH = "/home/nguyen.viet.anhd/Downloads/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/annotation/"
AU_PATH = "/home/nguyen.viet.anhd/Downloads/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio/"
for name in os.listdir(AU_PATH):
    name = name.split(".")[0]
    au, sr = librosa.load(AU_PATH + name + ".wav", 6000)
    anno = np.genfromtxt(open(ANNO_PATH + name + ".txt", "rb"))
    for row in anno:
        start, end, cr, wh = row
        frames = au[int(start*sr): int(end*sr)+1]
        if len(frames) < 512:
            frames = np.append(frames, [0]*(512-len(frames)))
        frames = librosa.util.frame(frames, 512, 256)
        frames = np.transpose(frames)
        # x = np.zeros((23 - len(frames) % 23, 512))
        # frames = np.append(frames, x, axis=0)
        count = 1
        f = np.log(librosa.feature.melspectrogram(frames[0], sr, n_fft=512, hop_length=1000, n_mels=60) + 1e-10)
        spec = []
        for i in range(0, len(frames)):
            spec.append(np.log(librosa.feature.melspectrogram(frames[i], sr, n_fft=512, hop_length=1000, n_mels=60) + 1e-10))
        spec = np.array(spec)[:, :, 0]
        n = 0
        while n < len(spec) - len(spec)%23:
            s = np.transpose(spec[n:n+23, :])
            n += 23
            if cr == 0 and wh == 0:
                # if np.random.rand() < 0.25:
                y.append(0)
                # print("Normal")
                c1 += 1
                # else:
                #     continue
            elif cr == 0 and wh == 1:
                # print("Wheeze")
                y.append(1)
                c2 += 1
            elif cr == 1 and wh == 0:
                # if np.random.rand() < 0.5:
                    # print("Crackle")
                y.append(2)
                c3 += 1
                # else:
                #     continue
            else:
                continue
            # plt.figure()
            # librosa.display.specshow(s)
            # plt.show()
            delta = librosa.feature.delta(s)
            s = np.expand_dims(s, 2)
            delta = np.expand_dims(delta, 2)
            X.append(np.append(s, delta, axis=2))
    print("Total:", len(X), len(y), X[0].shape, "Normal", c1, "Wheeze", c2, "Crackle", c3)

X = np.array(X)
y = np.array(y)
print("--------------")
print("Final data:", X.shape, y.shape)
print("Normal", c1, "Wheeze", c2, "Crackle", c3)
np.save("sliced_full.npy", X)
np.save("y_sliced_full.npy", y)
# X_new = []
# X = np.load("sliced_new.npy", allow_pickle=True)
# y = np.load("y_sliced.npy", allow_pickle=True)
# print(X.shape, y.shape)
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
