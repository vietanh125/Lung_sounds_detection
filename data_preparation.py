import tensorflow
import librosa
from scipy import signal
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import zscore
from keras.models import Model, Sequential
X = []
y = []
c1, c2, c3, c4 = 0, 0, 0, 0
ANNO_PATH = "/home/nguyen.viet.anhd/Downloads/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/annotation/"
AU_PATH = "/home/nguyen.viet.anhd/Downloads/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio/"
dir_ls = os.listdir(AU_PATH)



def feature_extraction(X, sample_rate):
    # Get features
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)  # 40 values
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, # tonal centroid features
                      axis=0)

    # Return computed features
    return [mfccs, chroma, mel, contrast, tonnetz, zcr]
for t in range(0, len(dir_ls)):
    name = dir_ls[t]
    name = name.split(".")[0]
    au, sr = librosa.load(AU_PATH + name + ".wav")
    anno = np.genfromtxt(open(ANNO_PATH + name + ".txt", "rb"))
    win_length = int(0.92*sr)
    for row in anno:
        start, end, cr, wh = row
        frames = au[int(start*sr): int(end*sr)+1]

        if len(frames) < win_length:
            frames = np.append(frames, [0]*(win_length-len(frames)))
        frames = librosa.util.frame(frames, win_length, int(win_length/2))
        frames = np.transpose(frames)
        # x = np.zeros((23 - len(frames) % 23, 512))
        # frames = np.append(frames, x, axis=0)
        count = 1
        f = np.log(librosa.feature.melspectrogram(frames[0], sr, n_fft=7600, hop_length=7600, n_mels=60) + 1e-10)
        # f = np.abs(librosa.core.stft(frames[0]))
        # spec = []
        for i in range(0, len(frames)):

            if cr == 0 and wh == 0:
                y.append(0)
                # print("Normal")
                c1 += 1
            elif cr == 0 and wh == 1:
                # print("Wheeze")
                y.append(1)
                c2 += 1
            elif cr == 1 and wh == 0:
                # print("Crackle")
                y.append(2)
                c3 += 1
            else:
                continue
            feat = feature_extraction(frames[i], sr)
            print(feat[1].shape, feat[2].shape, feat[3].shape, feat[4].shape, feat[5].shape, feat[0].shape,)
            X.append(feat)

            # spectro = np.abs(librosa.core.stft(frames[i], n_fft=256, hop_length=256))
            # print(spectro.shape)
            # X.append(zscore(spectro))
            # plt.figure()
            # db = librosa.amplitude_to_db(spectro)
            # librosa.display.specshow(db)
            # plt.title('Power spectrogram')
            # plt.colorbar(format='%+2.0f dB')
            # plt.tight_layout()
            # plt.show()
    print(t, "Total:", len(X), len(y), "Normal", c1, "Wheeze", c2, "Crackle", c3)
    # if t % 100 == 0:
    #     np.save("spec_" + str(t) + ".npy", np.array(X))
    #     np.save("y_spec_" + str(t) + ".npy", np.array(y))
    #     X = []
    #     y = []
X = np.array(X)
y = np.array(y)
print("--------------")
print("Final data:", X.shape, y.shape)
print("Normal", c1, "Wheeze", c2, "Crackle", c3)
np.save("spec_1.npy", X)
np.save("y_spec_1.npy", y)
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

