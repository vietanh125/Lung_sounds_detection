import tensorflow
import librosa
from scipy import signal
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from stockwell import st
from keras.applications.resnet50 import resnet50

from scipy.stats import zscore
from keras.models import Model, Sequential
X = []
y = []
c1, c2, c3, c4 = 0, 0, 0, 0
ANNO_PATH = "/home/nguyen.viet.anhd/Downloads/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/annotation/"
AU_PATH = "/home/nguyen.viet.anhd/Downloads/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio/"
dir_ls = os.listdir(AU_PATH)
import cv2
from speech_feature_extractor.gfcc_extractor import gfcc_extractor
from speech_feature_extractor.feature_extractor import cochleagram_extractor
import tensorflow as tf
from specAugment import spec_augment_tensorflow
# import data_preparation
from pydub import AudioSegment, playback
def gfcc_extract():
    global X, y, c1, c2, c3
    for t in range(0, len(dir_ls)):
        name = dir_ls[t]
        name = name.split(".")[0]
        frame, sr = librosa.load(AU_PATH + "107_3p2_Ar_mc_AKGC417L.wav")
        # frame = frame[0:1024]
        # print(librosa.feature.mfcc(frame, sr, n_mfcc=128).shape)
        # print(librosa.feature.chroma_cqt(frame, sr, n_chroma=128).shape)
        # print(librosa.feature.chroma_stft(frame, sr, n_fft=128).shape)


        # frame = data_preparation.butter_bandpass_filter(frame, 80.0, 1200.0, sr, 6).astype('float16')
        # sound = AudioSegment(frame.tobytes(), frame_rate=sr, sample_width=frame.dtype.itemsize, channels=1)
        # playback.play(sound)
        cochlea = cochleagram_extractor(frame, sr, 320, 160, 64, 'hanning')
        gfcc = gfcc_extractor(cochlea, 64, 32)
        gfcc = gfcc[:, 10:136]
        print(gfcc.shape)
        from time import time
        while (True):
            t = time()
            warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=gfcc, time_warping_para=5, time_masking_para=5, time_mask_num=6, frequency_mask_num=6, frequency_masking_para=5)
            print(time()-t)
            librosa.display.specshow(warped_masked_spectrogram)
            plt.show()

        print(warped_masked_spectrogram)
        # print(sd.query_devices())
        # print(sd.check_output_settings(device=5))
        # print(sr)
        # sd.play(au, sr, device=5)
        # sd.wait()
        # print("Done")
        anno = np.genfromtxt(open(ANNO_PATH + name + ".txt", "rb"))
        for row in anno:
            start, end, cr, wh = row
            frames = frame[int(start * sr): int(end * sr) + 1]
            cochlea = cochleagram_extractor(frames, sr, 320, 160, 64, 'hanning')
            gfcc = gfcc_extractor(cochlea, 64, 31)
gfcc_extract()
# X = np.array(X)
# y = np.array(y)
# print("--------------")
# print("Final data:", X.shape, y.shape)
# print("Normal", c1, "Wheeze", c2, "Crackle", c3)
# np.save("stft_full.npy", X)
# np.save("y_stft_full.npy", y)

# X = np.load("gfcc_32_126.npy", allow_pickle=True)
# y = np.load("y_gfcc_32_126.npy", allow_pickle=True)
# X_new = []
# y_new = []
# i = 0
# for i in range(0, len(X)):
#     if y[i] == 1:
#         for j in range(0, 3):
#             X_new.append(spec_augment_tensorflow.spec_augment(X[i], time_warping_para=20))
#             y_new.append(y[i])
#     if y[i] == 2:
#         X_new.append(spec_augment_tensorflow.spec_augment(X[i], time_warping_para=20))
#         y_new.append(y[i])
#     X_new.append(X[i])
#     y_new.append(y[i])
#
#     print(len(X_new))
# X = np.array(X_new)
# y = np.array(y_new)
# print("--------------")
# print("Final data:", X.shape, y.shape)
# print("Normal", len(y[y == 0]), "Wheeze", len(y[y == 1]), "Crackle", len(y[y == 2]))
# np.save("gfcc_aug_filter.npy", X)
# np.save("y_gfcc_aug_filter.npy", y)