import numpy as np
from pyclust import KMedoids
import numpy.random as rnd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from math import sin, cos, radians, pi, sqrt

# Creating data whose distribution seems two rings
def ring(r_min = 0, r_max = 1, n_samples = 360):
    angle = rnd.uniform(0, 2*pi, n_samples)
    distance = rnd.uniform(r_min, r_max, n_samples)
    data = []

    for a, d in zip(angle, distance):
        data.append([d*cos(a), d*sin(a)])
    
    return np.array(data)

data1 = ring(3, 5)
data2 = ring(24,27)

data = np.concatenate([data1, data2], axis=0) # axis vertical
labels = np.concatenate([[0 for i in range(0, len(data1))], [1 for i in range(0, len(data2))]])
plt.scatter(data[:,0], data[:,1], c=labels, s=5)
plt.title('Distributed data')
plt.show()

# Using KMeans method to see how it works with data distributions that looks like two rings
km = KMeans(2).fit(data)
clust = km.predict(data)
plt.scatter(data[:,0], data[:,1], c=clust, s=5)
plt.title('K-Means')
plt.show()

# KMedoides method
kmed = KMedoids(2).fit_predict(data)
plt.scatter(data[:,0], data[:,1], c=kmed)
plt.title('K-Medoides')
plt.show()

# Clustering Espectral
clust = SpectralClustering(2).fit_predict(data)
plt.scatter(data[:,0], data[:,1], c=clust, s=5)
plt.title('Clustering Espectral')
plt.show()

# NOTES
# ¿Podemos estimar k?
#       No: propagacion de la afinidad
#       Si: ¿Podemos usar la distancia euclidea?
#           Si: K-Means
#           No: ¿Buscar valores centrales?
#               Si: K-Medoides
#               No: ¿Los datos son linealmente separables?
#                   Si: Clustering aglomerativo (dendrograma)
#                   No: Clustering espectral