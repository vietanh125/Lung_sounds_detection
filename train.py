import numpy as np
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
from keras.applications.vgg19 import VGG19
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

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


X = np.load("bag_full.npy")
y = np.load("y_bag_full.npy")
i = 0
X_new = []
y_new = []
for i in range(0, len(y)):
    if y[i] == 0 and np.random.rand() > 0.06:
        continue
    elif y[i] == 1 and np.random.rand() > 0.25:
        continue
    elif y[i] == 2 and np.random.rand() > 0.125:
        continue
    X_new.append(X[i][0:194])
y = np.ones((len(X_new)))
X = np.append(np.array(X_new), np.load('bag_others.npy'), axis=0)
y = np.append(y, np.zeros(len(X) - len(y)))
print(X.shape, y.shape, '\n')
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
X = np.expand_dims(X, axis=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# base_model = VGG19(include_top=False, weights=None, input_shape=(60, 23, 2))
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.8)(x)
# x = Dense(units=3, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=x)

model = Sequential()
model.add(Conv1D(64, 3, input_shape=(194, 1)))
model.add(Activation('relu'))
model.add(Conv1D(64, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3))
model.add(Activation('relu'))
model.add(Conv1D(128, 3))
model.add(Activation('relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()
# opt = Adam(lr=0.01, momentum=0.9)
opt = SGD(lr=0.01, momentum=0.9, decay=1e-06, nesterov=True)
model.compile('adam', 'binary_crossentropy', metrics=[f1, 'accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=400, validation_data=(X_test, y_test))
y_pred = model.predict(X_test)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0

from plot_cf import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(y_test, y_pred, classes=['sound', 'non-sound'])
plt.show()