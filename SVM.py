from keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
import numpy as np

X = np.load("multi_features.npy")
y = np.load("y_multi.npy")
# X = X[0:1000, :]
# y = y[0:1000]
# y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# param = {'C': [0.0001, 0.01, 1, 100, 10000, 100000], 'gamma': [0.0001, 0.01, 1, 100, 10000, 100000]}
C = [0.0001, 0.01, 1, 100, 10000, 100000]
gamma = [0.0001, 0.01, 1, 100, 10000, 100000]
best_f1 = 0
best_c = 0
best_gamma = 0
for c in C:
    for g in gamma:
        svc = SVC(kernel='rbf', gamma=g, C=c)
        f1 = 0
        for i in range(0, 10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            svc.fit(X_train, y_train)
            y_pred = svc.predict(X_test)
            f1 += f1_score(y_test, y_pred, average='macro')
        f1 /= 10
        if f1 > best_f1:
            best_c = c
            best_f1 = f1
            best_gamma = gamma
        print("f1:", f1, "C:", c, "gamma", g)

print("best f1:", best_f1, "best_c:", best_c, "best_gamma:", best_gamma)

# grid = GridSearchCV(estimator=svc, cv=10, param_grid=param, scoring='f1_macro', n_jobs=-1, verbose=3)
# grid.fit(X_train, y_train)
