from sklearn.datasets._samples_generator import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

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

    # print(model.support_vectors_)
    if plot_support:
        ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=300, linewidth=1, facecolors='black')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

X, Y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2) # Example with semi-mixed points
plt.scatter(X[:,0], X[:,1], c=Y, s=50)
plt.show()

plt.scatter(X[:,0], X[:,1], c=Y, s=50)
model = SVC(kernel='linear', C=10)
model.fit(X,Y)
plt_svc(model) # Try to clasify the points
plt.show()

X, Y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8) # Another example with points less mixed
plt.scatter(X[:,0], X[:,1], c=Y, s=50)
plt.show()

fig, ax = plt.subplots(1,2, figsize=(16,6))
fig.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

for ax_i, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C)
    model.fit(X,Y)
    ax_i.scatter(X[:,0], X[:,1], c=Y, s=50)
    plt_svc(model, ax_i)
    ax_i.set_title(f'C = {C}', size=15)
plt.show() # Example to see the C parameter importance