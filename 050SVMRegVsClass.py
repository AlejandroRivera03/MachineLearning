import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

iris = datasets.load_iris()
# print(iris)

X = iris.data[:, :2]
Y = iris.target

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

h = (x_max - x_min)/100

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

X_plot = np.c_[xx.ravel(), yy.ravel()]

# Kernel lineal
C = 1.0
svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(X,Y)
Ypred = svc.predict(X_plot)
Ypred = Ypred.reshape(xx.shape)

plt.figure(figsize=(16,9))
plt.contourf(xx, yy, Ypred, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.tab10)
plt.xlabel('Longitud de los petalos')
plt.ylabel('Anchura de los petalos')
plt.xlim(xx.min(), xx.max())
plt.title('SVC para las flores de Iris con Kernel Lineal')
plt.show()

# Kernel radial
C = 1.0
svc = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr').fit(X,Y)
Ypred = svc.predict(X_plot)
Ypred = Ypred.reshape(xx.shape)

plt.figure(figsize=(16,9))
plt.contourf(xx, yy, Ypred, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.tab10)
plt.xlabel('Longitud de los petalos')
plt.ylabel('Anchura de los petalos')
plt.xlim(xx.min(), xx.max())
plt.title('SVC para las flores de Iris con Kernel Radial')
plt.show()

# Kernel sigmoide
C = 1.0
svc = svm.SVC(kernel='sigmoid', C=C, decision_function_shape='ovr').fit(X,Y)
Ypred = svc.predict(X_plot)
Ypred = Ypred.reshape(xx.shape)

plt.figure(figsize=(16,9))
plt.contourf(xx, yy, Ypred, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.tab10)
plt.xlabel('Longitud de los petalos')
plt.ylabel('Anchura de los petalos')
plt.xlim(xx.min(), xx.max())
plt.title('SVC para las flores de Iris con Kernel Sigmoide')
plt.show()

# Kernel sigmoide
C = 1.0
svc = svm.SVC(kernel='poly', C=C, decision_function_shape='ovr').fit(X,Y)
Ypred = svc.predict(X_plot)
Ypred = Ypred.reshape(xx.shape)

plt.figure(figsize=(16,9))
plt.contourf(xx, yy, Ypred, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.tab10)
plt.xlabel('Longitud de los petalos')
plt.ylabel('Anchura de los petalos')
plt.xlim(xx.min(), xx.max())
plt.title('SVC para las flores de Iris con Kernel Polynomial')
plt.show()

X, Y = shuffle(X, Y, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
parameters = [
    {
        'kernel': ['rbf'],
        'gamma': [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5],
        'C': [1, 10, 100, 1000]
    },
    {
        'kernel': ['linear'],
        'C': [1, 10, 100, 1000]
    }
]

clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), param_grid=parameters, cv=5)
clf.fit(X, Y)

print(f'Best params => {clf.best_params_}')
# print(clf.cv_results_)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
params = clf.cv_results_['params']

for m, s, p in zip(means, stds, params):
    print(f'{m} (+/- {2*s}) para {p}')


y_pred = clf.predict(X_test)
print(classification_report(Y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))