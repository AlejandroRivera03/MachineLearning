# Support Verctor Machine

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.svm import SVC
from sklearn.datasets._samples_generator import make_blobs

X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap='autumn')
plt.plot([0.5], [2.1], 'x', color='blue', markeredgewidth=2, markersize=10) # Where does this point belong to?
plt.show()

# Some possible models
plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap='autumn')
plt.plot([0.5], [2.1], 'x', color='blue', markeredgewidth=2, markersize=10)
xx = np.linspace(-1, 3.5)
for a, b in [(1, 0.65), (0.5, 1.6,), (-0.2, 2.9)]:
    yy = a*xx + b
    plt.plot(xx, yy, 'k-')

plt.xlim(-1, 3.5)
plt.show()

# Maximizacion de margen para seleccion de modelo (el mas ancho)
plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap='autumn')
plt.plot([0.5], [2.1], 'x', color='blue', markeredgewidth=2, markersize=10)
xx = np.linspace(-1, 3.5)
for a, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yy = a*xx + b
    plt.plot(xx, yy, 'k-')
    plt.fill_between(xx, yy-d, yy+d, edgecolor='none', color='#BBBBBB', alpha=0.4)

plt.xlim(-1, 3.5)
plt.show()

#

model = SVC(kernel='linear', C=1E10)
model.fit(X,Y)

def plt_svc(model, ax=None, plot_support=True):
    """Plot de la funcion de decision para una clasificacion en 2D con SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Generamos la parrila de puntos para evaluar el model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(yy, xx)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # Representamos las fronteras y los margenes del SVC
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    print(model.support_vectors_)
    if plot_support:
        ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=300, linewidth=1, facecolors='black')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap='autumn')
plt_svc(model, plot_support=True)
plt.show()

def plot_svm(N=10, ax=None):
    X, Y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.6)

    X = X[:N]
    Y = Y[:N]
    model = SVC(kernel='linear', C=1E10)
    model.fit(X,Y)

    ax = ax or plt.gca()
    ax.scatter(X[:,0], X[:,1], c=Y, s=50, cmap='autumn')
    ax.set_xlim(-1,4)
    ax.set_ylim(-1,6)
    plt_svc(model, ax)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for ax_i, N in zip(ax, [60, 120]):
    plot_svm(N, ax_i)
    ax_i.set_title(f'N={N}')
plt.show()