import tensorflow
import librosa
from scipy import signal
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from stockwell import st
from keras.applications.resnet50 import resnet50
from scipy.signal import butter, lfilter
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.stats import zscore
from keras.models import Model, Sequential
from specAugment.spec_augment_tensorflow import spec_augment
X = []
y = []
c1, c2, c3, c4 = 0, 0, 0, 0
ANNO_PATH = "/home/nguyen.viet.anhd/Downloads/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/annotation/"
AU_PATH = "/home/nguyen.viet.anhd/Downloads/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio/"
dir_ls = os.listdir(AU_PATH)
from pydub.playback import play
# play(audio_segment)


from speech_feature_extractor.gfcc_extractor import gfcc_extractor
from speech_feature_extractor.feature_extractor import cochleagram_extractor
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def norm(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def feature_extraction(X, sample_rate):
    # Get features
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0).flatten()  # 40 values
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X)).flatten()
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0).flatten()
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0).flatten()
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0).flatten()
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, # tonal centroid features
                      axis=0).flatten()
    cochlea = cochleagram_extractor(X, sample_rate, 320, 160, 64, 'hanning')
    gfcc = np.mean(gfcc_extractor(cochlea, 64, 31).T, axis=0).flatten()
    cochlea = np.mean(cochlea.T, axis=0).flatten()
    # Return computed features
    return np.concatenate((mfccs, chroma, mel, contrast, tonnetz, zcr, gfcc), axis=0)

def bag_others():
    global X, y, c1, c2, c3
    for t in range(0, len(dir_ls)):
        name = dir_ls[t]
        name = name.split(".")[0]
        au, sr = librosa.load(AU_PATH + name + ".wav")
        anno = np.genfromtxt(open(ANNO_PATH + name + ".txt", "rb"))
        win_length = int(0.92 * sr)
        s = 0
        for i in range(0, len(anno) + 1):
            if i < len(anno):
                start, end, cr, wh = anno[i]
            else:
                start = (len(au)-1)/sr
                end = start
            frames = au[int(s * sr): int(start * sr) + 1]
            s = end
            if len(frames) < win_length:
                if len(frames) <= 1000:
                    continue
                frames = np.append(frames, [0] * (win_length - len(frames)))
            frames = librosa.util.frame(frames, win_length, int(win_length / 2))
            frames = np.transpose(frames)
            for f in frames:
                feat = feature_extraction(f, sr)
                X.append(feat)
        print(t, "Total:", len(X), X[0].shape, "Normal", c1, "Wheeze", c2, "Crackle", c3)
        # if len(X) >= 3000 and t != 0:
        #     np.save("stft_" + str(t) + ".npy", np.array(X))
        #     np.save("y_stft_" + str(t) + ".npy", np.array(y))
        #     X = []
        #     y = []

def bag():

    global X, y, c1, c2, c3
    for t in range(0, len(dir_ls)):
        name = dir_ls[t]
        name = name.split(".")[0]
        # if name.split("_")[4] != "Litt3200" and name.split("_")[4] != "Meditron":
        #     continue
        au, sr = librosa.load(AU_PATH + name + ".wav")
        anno = np.genfromtxt(open(ANNO_PATH + name + ".txt", "rb"))
        win_length = int(0.92 * sr)
        for row in anno:
            start, end, cr, wh = row
            frames = au[int(start * sr): int(end * sr) + 1]
            frames = butter_bandpass_filter(frames, 80.0, 1800.0, sr, 4)

            if len(frames) < win_length:
                frames = np.append(frames, [0]*(win_length-len(frames)))
            frames = librosa.util.frame(frames, win_length, int(win_length/2))
            frames = np.transpose(frames)
            for f in frames:
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
                feat = feature_extraction(f, sr)
                print(feat)
                X.append(feat)
        print(t, "Total:", len(X), X[0].shape, "Normal", c1, "Wheeze", c2, "Crackle", c3)
        # if len(X) >= 3000 and t != 0:
        #     np.save("stft_" + str(t) + ".npy", np.array(X))
        #     np.save("y_stft_" + str(t) + ".npy", np.array(y))
        #     X = []
        #     y = []

def gfcc_others():
    global X, y, c1, c2, c3
    for t in range(0, len(dir_ls)):
        name = dir_ls[t]
        name = name.split(".")[0]
        # if name.split("_")[4] != "Litt3200" and name.split("_")[4] != "Meditron":
        #     continue
        au, sr = librosa.load(AU_PATH + name + ".wav")
        anno = np.genfromtxt(open(ANNO_PATH + name + ".txt", "rb"))
        s = 0
        for i in range(0, len(anno) + 1):
            if i < len(anno):
                start, end, cr, wh = anno[i]
            else:
                start = (len(au) - 1) / sr
                end = start
            frames = au[int(s * sr): int(start * sr) + 1]
            s = end
            if len(frames) <= 1000:
                continue
            cochlea = cochleagram_extractor(frames, sr, 320, 160, 64, 'hanning')
            gfcc = gfcc_extractor(cochlea, 64, 32)
            j = 0
            while j < gfcc.shape[1]:
                feat = gfcc[:, int(j): min(int(j + 126), gfcc.shape[1])]
                feat = np.append(feat, np.zeros((feat.shape[0], 126 - feat.shape[1])), axis=1)
                X.append(feat)
                j += 63
        print(t, "Total:", len(X), X[0].shape, "Normal", c1, "Wheeze", c2, "Crackle", c3)
        # if len(X) >= 3000 and t != 0:
        #     np.save("stft_" + str(t) + ".npy", np.array(X))
        #     np.save("y_stft_" + str(t) + ".npy", np.array(y))
        #     X = []
        #     y = []
from pydub import AudioSegment
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)
def gfcc_extract():
    global X, y, c1, c2, c3
    for t in range(0, len(dir_ls)):
        name = dir_ls[t]
        name = name.split(".")[0]
        # if name.split("_")[4] != "Litt3200" and name.split("_")[4] != "Meditron":
        #     continue
        au, sr = librosa.load(AU_PATH + name + ".wav")
        anno = np.genfromtxt(open(ANNO_PATH + name + ".txt", "rb"))
        for row in anno:
            start, end, cr, wh = row
            if cr == 1 and wh == 1:
                continue
            frames = au[int(start * sr): int(end * sr) + 1]
            # band-pass
            frames = butter_bandpass_filter(frames, 80.0, 1200.0, sr, 4).astype('float32')

            # normalize volume
            # sound = AudioSegment(frames.tobytes(), frame_rate=sr, sample_width=frames.dtype.itemsize, channels=1)
            # normalized_sound = match_target_amplitude(sound, -20.0)
            # frames = np.array(normalized_sound.get_array_of_samples())

            cochlea = cochleagram_extractor(frames, sr, 320, 160, 64, 'hanning')
            gfcc = gfcc_extractor(cochlea, 64, 32)
            # librosa.display.specshow(gfcc)
            # plt.show()
            i = 0
            while i + 126 < gfcc.shape[1]:
                feat = gfcc[:, int(i): int(i + 126)]
                feat = np.append(feat, np.zeros((feat.shape[0], 126 - feat.shape[1])), axis=1)
                print("----------> ", c1, c2, c3)
                if cr == 0 and wh == 0:
                    X.append(feat)
                    y.append(0)
                    c1 += 1
                elif cr == 0 and wh == 1:
                    for j1 in range(0, 4):
                        aug = spec_augment(mel_spectrogram=feat, time_warping_para=20)
                        X.append(aug)
                        y.append(1)
                        c2 += 1
                elif cr == 1 and wh == 0:
                    for j2 in range(0, 2):
                        aug = spec_augment(mel_spectrogram=feat, time_warping_para=20)
                        X.append(aug)
                        y.append(2)
                        c3 += 1
                i += 63
        print(t, "Total:", len(X), X[0].shape, "Normal", c1, "Wheeze", c2, "Crackle", c3)
        if len(X) >= 10000 and t != 0:
            np.save("gfcc_aug_filter_" + str(t) + ".npy", np.array(X))
            np.save("y_gfcc_aug_filter_" + str(t) + ".npy", np.array(y))
            X = []
            y = []
gfcc_extract()
X = np.array(X)
y = np.array(y)
print("--------------")
print("Final data:", X.shape, y.shape)
print("Normal", c1, "Wheeze", c2, "Crackle", c3)
np.save("gfcc_aug_filter.npy", X)
np.save("y_gfcc_aug_filter.npy", y)

