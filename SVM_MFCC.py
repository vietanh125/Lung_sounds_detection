from keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

X = np.load("bag.npy")
y = np.load("y_bag.npy")
X = X[:, 0:40]
print(X.shape, y.shape, '\n')
# X = X[0:1000, :]
# y = y[0:1000]
# y = to_categorical(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# param = {'C': [0.0001, 0.01, 1, 100, 10000, 100000], 'gamma': [0.0001, 0.01, 1, 100, 10000, 100000]}
C = [1000]
gamma = [0.0001]
best_f1 = 0.8356317934782608
best_c = 1000
best_gamma = 0.0001
for c in C:
    for g in gamma:
        svc = SVC(kernel='rbf', gamma=g, C=c)
        f1 = 0
        for i in range(0, 10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
            svc.fit(X_train, y_train)
            y_pred = svc.predict(X_test)
            score = f1_score(y_test, y_pred, average='macro')
            f1 += score
            print(score)
        f1 /= 10
        if f1 > best_f1:
            best_c = c
            best_f1 = f1
            best_gamma = g
        print("f1:", f1, "C:", c, "gamma", g)
    print('\n')

print("best f1:", best_f1, "best_c:", best_c, "best_gamma:", best_gamma)

# grid = GridSearchCV(estimator=svc, cv=10, param_grid=param, scoring='f1_macro', n_jobs=-1, verbose=3)
# grid.fit(X_train, y_train)
