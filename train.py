import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
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


X = np.load("sliced_new.npy", allow_pickle=True)
y = np.load("y_sliced.npy", allow_pickle=True)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


model = Sequential()
model.add(Convolution2D(64, 3, padding='same', input_shape=(60, 23, 2), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(2, 2, padding='same'))

model.add(Convolution2D(128, 3, padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(128, 3, padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(2, 2, padding='same'))

model.add(Convolution2D(256, 3, padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Convolution2D(256, 3, padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(2, 2, padding='same'))

model.add(Flatten())
model.add(Dense(1024, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1024, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(3, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('softmax'))

opt = SGD(lr=0.01, momentum=0.9)
model.compile(opt, 'categorical_crossentropy', metrics=[f1])
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test))