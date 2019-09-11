from matplotlib.animation import FuncAnimation
from audiolazy import lazy_lpc as lpc
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import sys
from pydub import AudioSegment
from collections import deque
import librosa
import keras.backend as K
from keras.models import load_model
import data_preparation
import noisereduce as nr

from speech_feature_extractor.gfcc_extractor import gfcc_extractor
from speech_feature_extractor.feature_extractor import cochleagram_extractor
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
classifier = load_model("vgg-95-0.88.h5", custom_objects={'f1':f1})
# vgg-30-0.65.h5: wheeze and crackle
# vgg-95-0.88.h5 probably the best
import pickle
classifier.summary()
detector = pickle.load(open("svm_detector.pkl", 'rb'))
#############################
# GUI parameters
#############################
headless = False
timeDomain = True
freqDomain = False
lpcOverlay = False

#############################
# Stream Parameters
#############################
DEVICE = 1
CHUNK = 4080
WINDOW = np.hamming(CHUNK)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050

#############################
# Spectral parameters
#############################
ORDER = 12
NFFT = CHUNK*2
# correlation: 12*4*1024 = 49152
# matrix solution: 12*12 = 144
# matrix solution cov: 12*12*12 = 1728

from pydub import playback
#############################
# Feature extraction
#############################
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def est_predictor_gain(x, a):
    cor = np.correlate(x, x, mode='full')
    rr = cor[int(len(cor)/2): int(len(cor)/2)+ORDER+1]
    g = np.sqrt(np.sum(a*rr))
    return g


def lpc_spectrum(data):
    a = lpc.lpc.autocor(data, ORDER)
    g = est_predictor_gain(data, a.numerator)
    spectral_lpc = np.fft.fft([xx/g for xx in a.numerator], NFFT)
    S = -20*np.log10(np.abs(spectral_lpc)**2)
    return S[0:int(NFFT/2)]


def spectral_estimate(data):
    spectral = np.fft.fft(data, NFFT)
    S = 20*np.log10(np.abs(spectral)**2)
    return S[0:int(NFFT/2)]

##########################
# Create audio stream
##########################
p = pyaudio.PyAudio()
print ("Checking compatability with input parameters:")
print ("\tAudio Device:", DEVICE)
print ("\tRate:", RATE)
print ("\tChannels:", CHANNELS)
print ("\tFormat:", FORMAT)

# isSupported = p.is_format_supported(input_format=FORMAT,
#                                     input_channels=1,
#                                     rate=RATE,
#                                     input_device=DEVICE)
# if isSupported:
#     print ("\nThese settings are supported on device %i!\n" % (DEVICE))
# else:
#     sys.exit("\nUh oh, these settings aren't",
#              " supported on device %i.\n" % (DEVICE))

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

##########################
# Fetch incoming data
##########################
errorCount = [0]
buff = deque(np.zeros((int(20400/CHUNK), CHUNK)), maxlen=int(20400/CHUNK))
import time
if not headless:
    oldy = range(10200)

    ######################################################
    # Create new Figure and an Axes to occupy the figure.
    ######################################################
    fig = plt.figure(facecolor='white')  # optional arg: figsize=(x, y)
    nAx = sum([1 for xx in [timeDomain, freqDomain] if xx])

    axTime = fig.add_axes([.1, .5*nAx - .5 + .1/nAx, .8, .4*(3-nAx)])
    axTime.set_xticklabels('#c6dbef')
    plt.grid()
    lineTime, = axTime.plot(range(20400), range(20400), c='#08519c')
    plt.ylim([-80000, 80000])
    plt.xlim([0, 20400])
    plt.title('Real Time Audio (milliseconds)')
    axTime.set_xticks(np.linspace(0, 20400, 5))
    labels = ['%.1f' % (xx) for xx in np.linspace(0, 1000*20400/RATE, 5)]
    axTime.set_xticklabels(labels, rotation=0, verticalalignment='top')
    plt.ylabel('Amplitude')
    text = axTime.text(0.25, 0.05, "", dict(size=15), transform=axTime.transAxes, color='green')

    noise = np.fromstring(stream.read(CHUNK*2), np.float16)
    noise = data_preparation.butter_bandpass_filter(noise, 80.0, 1200.0, RATE, 4).astype('float32')

    ######################################################
    # Define function to update figure for each iteration.
    ######################################################
    for i in range(0, len(buff)-1):
        buff.append(np.fromstring(stream.read(CHUNK), np.int16))
    flag = True
    noise = None
    def update(frame_number):
        global oldy, flag, noise
        # global first, sec, third
        global axTime
        objects_to_return = []
        try:
            incoming = np.fromstring(stream.read(CHUNK), np.int16)
            if timeDomain:
                buff.append(incoming)
                frame = np.array(buff).flatten()
                buff.popleft()
                # frame = data_preparation.butter_bandpass_filter(frame, 80.0, 1200.0, RATE, 4).astype('float16')
                newy = list(oldy[20400:])
                newy.extend(frame)
                lineTime.set_ydata(newy)
                objects_to_return.append(lineTime)
                # frame = np.array(normalized_sound.get_array_of_samples()).astype('float16')
                if flag:
                    noise = frame
                    flag = False
                    print("flag")
                # frame = nr.reduce_noise(audio_clip=frame, noise_clip=noise, verbose=False).astype('float16')
                sound = AudioSegment(frame.tobytes(), frame_rate=RATE, sample_width=frame.dtype.itemsize, channels=1)
                playback.play(sound)

                thresh = np.sum(abs(frame))/len(frame)
                if thresh > 0:
                    cochlea = cochleagram_extractor(frame, RATE, 320, 160, 64, 'hanning')
                    gfcc = gfcc_extractor(cochlea, 64, 32)
                    # print(np.mean(gfcc), np.std(gfcc))
                    # svm_feat = data_preparation.feature_extraction(frame/1.0, RATE)
                    # print(detector.predict([svm_feat]))
                    gfcc = np.expand_dims(gfcc, 2)
                    gfcc = np.expand_dims(gfcc, 0)

                    # svm_feat = data_preparation.feature_extraction(frame/1.0, RATE)
                    # svm_feat = np.expand_dims(svm_feat, 0)
                    # svm_feat = np.expand_dims(svm_feat, 2)
                    res = classifier.predict(gfcc)
                    pred = np.argmax(res)
                    if pred == 0:
                        st = "normal"
                    elif pred == 1:
                        st = "wheeze"
                    else:
                        st = "crackle"
                    acc = res[0][pred]
                    text.set_text("predict: " + st + " - acc:" + str(acc))
                else:
                    text.set_text("others")
                # first = sec
                # sec = third
                # third = incoming
                objects_to_return.append(text)
                oldy = newy
        except IOError as e:
            print (str(e))
            errorCount[0] += 1
            if errorCount[0] > 5:
                print ("This is happening often.")
                print (" (Try increasing size of \'CHUNK\'.)")
            else:
                pass
        return objects_to_return

    animation = FuncAnimation(fig, update, interval=10, blit=True)
    plt.show()

else:
    first = np.fromstring(stream.read(CHUNK), np.int16)
    sec = np.fromstring(stream.read(CHUNK), np.int16)
    third = np.fromstring(stream.read(CHUNK), np.int16)
    while True:
        incoming = np.fromstring(stream.read(CHUNK), np.int16)
        t = time.time()
        frame = np.concatenate((first, sec, third, incoming), axis=0)
        # cochlea = cochleagram_extractor(frame, RATE, 320, 160, 126, 'hanning')
        # gfcc = gfcc_extractor(cochlea, 126, 32)
        svm_feat = data_preparation.feature_extraction(frame, RATE)
        # print(detector.predict([svm_feat]))
        # gfcc = np.expand_dims(gfcc, 2)
        # gfcc = np.expand_dims(gfcc, 0)
        svm_feat = np.expand_dims(svm_feat, 0)
        svm_feat = np.expand_dims(svm_feat, 2)

        res = classifier.predict(svm_feat)
        pred = np.argmax(res)
        acc = res[0][pred]
        first = sec
        sec = third
        third = incoming
        print("predict", pred, acc)
