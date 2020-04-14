# X => dataset (array n * m) de puntos a clusterizar
# n => numero de datos
# m numero de rasgos
# Z => array de enlace del cluster con la informacion de las uniones
# k => numero de clusters

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import numpy as np

np.random.seed(4711)
a = np.random.multivariate_normal([10,0], [[3,1], [1,4]], size=[100,])
b = np.random.multivariate_normal([0,20], [[3,1], [1,4]], size=[50,])
X = np.concatenate((a,b))
print(X.shape)

plt.scatter(X[:,0], X[:,1])
plt.show()

Z = linkage(X, 'ward')

c, coph_dist = cophenet(Z, pdist(X))
print(f'Precision en conservacion de las distancias originales con respecto a los clusters generados => {c}')

idx = [33,62,68] # Clusters that were joined
idx2 = [15, 69, 41] # Clusters that were joined
plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1]) # Displaying all points
plt.scatter(X[idx, 0], X[idx,1], c='r') # Displaying a group of points that were joined
plt.scatter(X[idx2, 0], X[idx2,1], c='y') # Displaying a group of points that were joined
plt.show()