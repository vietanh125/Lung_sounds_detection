from keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

X = np.load("bag_full.npy")
y = np.load("y_bag_full.npy")
i = 0
X_new = []
y_new = []
X_temp = []
for i in range(0, len(y)):
    if y[i] == 0 and np.random.rand() > 0.06:
        X_temp.append(X[i][0:194])
        continue
    elif y[i] == 1 and np.random.rand() > 0.25:
        X_temp.append(X[i][0:194])
        continue
    elif y[i] == 2 and np.random.rand() > 0.125:
        X_temp.append(X[i][0:194])
        continue
    X_new.append(X[i][0:194])
y = np.ones((len(X_new)))
X = np.append(np.array(X_new), np.load('bag_others.npy'), axis=0)
y = np.append(y, np.zeros(len(X) - len(y)))
print(X.shape, y.shape, '\n')
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_test = np.append(X_test, np.array(X_temp), axis=0)
y_test = np.append(y_test, np.ones(len(X_temp)))
print(X_train.shape, X_test.shape, '\n')
print(y_train[y_train==0].shape)
param = {'C': [13], 'gamma': [1], 'kernel': ['rbf']}

svc = SVC()
# svc.fit(X_train, y_train)
grid = GridSearchCV(estimator=svc, cv=5, param_grid=param, scoring='accuracy', n_jobs=4, verbose=2)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
y_pred = grid.best_estimator_.predict(X_test)
res = f1_score(y_test, y_pred, average='macro')
acc = accuracy_score(y_test, y_pred)
print(res, acc)
from plot_cf import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(y_test, y_pred, classes=['sound', 'non-sound'])
plt.show()
import pickle
pickle.dump(grid.best_estimator_, open("svm_detector.pkl", 'wb'))
