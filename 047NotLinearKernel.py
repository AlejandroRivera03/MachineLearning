from sklearn.svm import SVC
from mpl_toolkits import mplot3d
from sklearn.datasets._samples_generator import make_circles
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


X, Y = make_circles(100, factor=0.1, noise=0.1)
r = np.exp(-(X**2).sum(1)) # Data to become the points in #3D giving them a deep value
print(r)

def plt_3D(elev=30, azim=30, X=X, Y=Y, r=r):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:,0], X[:,1], r, c=Y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_zlabel('r')

plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap='autumn') # Points
plt.show()

plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap='autumn') # Tryinh to classify the point by a rect
plt_svc(SVC(kernel='linear').fit(X,Y), plot_support=False)
plt.show()

plt_3D(X=X, Y=Y, r=r)
plt.show()