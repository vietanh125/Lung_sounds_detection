import numpy as np
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import keras.backend as K
import matplotlib.pyplot as plt
import librosa
import librosa.display
import cv2
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

# X = np.load("spec_1.npy", allow_pickle=True)
# # y = np.load("y_spec_1.npy", allow_pickle=True)
# X_new = []
# for x in X:
#     flat = np.append(np.append(np.append(np.append(np.append(x[0].flatten(), x[1].flatten()), x[2].flatten()), x[3].flatten()), x[4].flatten()), x[5].flatten())
#     print(flat.shape)
#     X_new.append(flat)
#
# np.save("multi_features.npy", np.array(X_new))
X = np.load("multi_features.npy", allow_pickle=True)
y = np.load("y_multi.npy", allow_pickle=True)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
base_model = InceptionResNetV2(include_top=False, weights=None, input_shape=(60, 23, 2))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.8)(x)
x = Dense(units=3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)
# model = Sequential()
# model.add(Convolution2D(64, 3, padding='same', input_shape=(60, 23, 2), use_bias=False, kernel_initializer=''))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Convolution2D(64, 3, padding='same', use_bias=False))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPool2D(2, 2, padding='same'))
#
# model.add(Convolution2D(128, 3, padding='same', use_bias=False))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Convolution2D(128, 3, padding='same', use_bias=False))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPool2D(2, 2, padding='same'))
#
# # model.add(Convolution2D(256, 3, padding='same', use_bias=False))
# # model.add(BatchNormalization())
# # model.add(Activation('relu'))
# #
# # model.add(Convolution2D(256, 3, padding='same', use_bias=False))
# # model.add(BatchNormalization())
# # model.add(Activation('relu'))
# # model.add(MaxPool2D(2, 2, padding='same'))

# model.add(Flatten())
# model.add(Dense(1024, use_bias=False))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Dropout(0.5))
#
# model.add(Dense(1024, use_bias=False))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
#
# model.add(Dropout(0.5))
#
# model.add(Dense(3, use_bias=False))
# model.add(BatchNormalization())
# model.add(Activation('softmax'))

opt = Adam(lr=0.01, momentum=0.9)
model.compile(opt, 'categorical_crossentropy', metrics=[f1])
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))