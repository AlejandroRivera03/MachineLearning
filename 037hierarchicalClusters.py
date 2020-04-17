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

# DENDROGRAMA
plt.figure(figsize=(19,9))
plt.title('Dendrograma del clustering jerárquico')
plt.xlabel('Índice de la Muestra')
plt.ylabel('Distancias')
dendrogram(Z, leaf_rotation=90.0, leaf_font_size=8.0)
plt.show()

print(Z[-5:]) # Last clusters' unions

# DENDROGRAMA TRUNCADO
plt.figure(figsize=(19,9))
plt.title('Dendrograma del clustering jerárquico')
plt.xlabel('Índice de la Muestra')
plt.ylabel('Distancias')
dendrogram(Z, leaf_rotation=90.0, leaf_font_size=8.0, truncate_mode='lastp', p=12, show_leaf_counts=True, show_contracted=True)
plt.show()

# DENDROGRAMA TUNEADO

def dendrogram_tune(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d # where to separate clusters
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Clustering Jerarquico con Dendrograma Truncado y Tuneado')
        plt.xlabel('Índice del dataset (o tamaño del cluster)')
        plt.ylabel('Distancia')
        for i, dist, color in zip( ddata['icoord'], ddata['dcoord'], ddata['color_list'] ):
            print(i, dist, color)
            x = 0.5 * sum(i[1:3])
            y = dist[1]
            print(x, y)
            if y > annotate_above:
                plt.plot(x, y, 'o', c=color)
                plt.annotate('%.3g'%y, (x,y), xytext=(0,-5), textcoords='offset points', va='top', ha='center')
        
    if max_d:
        plt.axhline(y=max_d, c='k')
    
    return ddata

# max_d => y = max_d, it displays an horizontal line to cut the dendrogram (max distance)
# annotate_above => distances unions more than annotate_above are emphasized
# The other properties are dendrogram parameters
dendrogram_tune(Z, truncate_mode='lastp', p=12, leaf_rotation=90.0, leaf_font_size=12.0, show_contracted=True, annotate_above=15, max_d=20)
plt.show()