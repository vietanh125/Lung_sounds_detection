from matplotlib.animation import FuncAnimation
from audiolazy import lazy_lpc as lpc
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import sys
from keras.models import load_model
import keras.backend as K

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

clf = load_model("weights-improvement-186-0.81.h5", custom_objects={'f1': f1})
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
DEVICE = 5
CHUNK = 10200
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


#############################
# Feature extraction
#############################
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
if not headless:
    oldy = range(2*CHUNK)

    ######################################################
    # Create new Figure and an Axes to occupy the figure.
    ######################################################
    fig = plt.figure(facecolor='white')  # optional arg: figsize=(x, y)
    nAx = sum([1 for xx in [timeDomain, freqDomain] if xx])

    if timeDomain:
        axTime = fig.add_axes([.1, .5*nAx - .5 + .1/nAx, .8, .4*(3-nAx)])
        axTime.set_xticklabels('#c6dbef')
        plt.grid()
        lineTime, = axTime.plot(range(2*CHUNK), range(2*CHUNK), c='#08519c')
        plt.ylim([-30000, 30000])
        plt.xlim([0, 2*CHUNK])
        plt.title('Real Time Audio (milliseconds)')
        axTime.set_xticks(np.linspace(0, 2*CHUNK, 5))
        labels = ['%.1f' % (xx) for xx in np.linspace(0, 1000*2*CHUNK/RATE, 5)]
        axTime.set_xticklabels(labels, rotation=0, verticalalignment='top')
        plt.ylabel('Amplitude')

    if freqDomain:
        axFreq = fig.add_axes([.1, .1/nAx, .8, 0.4*(3-nAx)])
        axFreq.set_xticklabels('#c6dbef')
        plt.grid()
        lineFreq, = axFreq.plot(range(int(NFFT/2)), [0]*(int(NFFT/2)),
                                c='#6baed6', label='FFT')
        if lpcOverlay:
            lineLPC, = axFreq.plot(range(int(NFFT/2)), [0]*(int(NFFT/2)),
                                   '--', c='#08519c', label='LPC Envelope')
        plt.ylim([-50, 300])
        plt.xlim([0, NFFT/2])
        plt.title('Real Time Audio (Hz)')
        plt.legend()
        axFreq.set_xticks(np.linspace(0, NFFT/2, 5))
        labels = ['%.1f' % (xx) for xx in np.linspace(0, RATE/2, 5)]
        axFreq.set_xticklabels(labels, rotation=0, verticalalignment='top')
        plt.ylabel('Amplitude [dB]')

    ######################################################
    # Define function to update figure for each iteration.
    ######################################################
    def update(frame_number):
        global oldy
        objects_to_return = []
        try:
            incoming = np.fromstring(stream.read(CHUNK), 'Int16')
            if timeDomain:
                newy = list(oldy[CHUNK:])
                newy.extend(incoming)
                lineTime.set_ydata(newy)
                objects_to_return.append(lineTime)
                oldy = newy
            if freqDomain:
                windowed = incoming*WINDOW
                S = spectral_estimate(windowed)
                lineFreq.set_ydata(S)
                if lpcOverlay:
                    S_lpc = lpc_spectrum(windowed)
                    lineLPC.set_ydata(S_lpc)
                objects_to_return.append(lineFreq)
                objects_to_return.append(lineLPC)
        except IOError as e:
            print (str(e))
            errorCount[0] += 1
            if errorCount[0] > 5:
                print ("This is happening often.")
                print (" (Try increasing size of \'CHUNK\'.)")
            else:
                pass
        return objects_to_return

    animation = FuncAnimation(fig, update, interval=10)
    plt.show()

else:
    while True:
        incoming = np.fromstring(stream.read(CHUNK), 'Int16')
        S = spectral_estimate(windowed)
        #  Do some processing stuff here!