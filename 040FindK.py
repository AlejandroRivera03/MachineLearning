import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score

x1 = np.array([3,1,1,2,1,6,6,6,5,6,7,8,9,8,9,9,8])
x2 = np.array([5,4,5,6,5,8,6,7,6,7,1,2,1,2,3,2,3])
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

plt.plot()
plt.xlim([0,10]) # Limites
plt.ylim([0,10])
plt.title('Dataset a clasificar')
plt.scatter(x1,x2)
plt.show()

max_k = 10 # Numero maximo de cluster
K = range(1, max_k)
ssw = [] # Suma de los cuadrados internos
color_palette = [plt.cm.nipy_spectral(float(i)/max_k) for i in K]
# print(color_palette)
centroid = [sum(X)/len(X) for i in K]
sst = sum(np.min(cdist(X, centroid, 'euclidean'), axis=1)) # Suma de los cuadrados totales

for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)

    centers = pd.DataFrame(kmeanModel.cluster_centers_)
    labels = kmeanModel.labels_

    ssw_k = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1))
    ssw.append(ssw_k)

    label_color = [color_palette[i] for i in labels]

    # Fabricaremos una silueta para cada cluster
    # Por seguridad, no hacemos silueta si k = 1 (un solo cluster) o k = len(K) (un cluster para cada cluster individual)
    if 1<k<len(K):
        # Crear un subplot en una fila y dos columnas
        fig, (axis1, axis2) = plt.subplots(1,2)
        fig.set_size_inches(18,8)

        # El primer subplot contendra la silueta, que puede tener valores desde -1 a 1
        # En nuestro caso, ya controlamos que los valores estan entre -0.1 y 1
        axis1.set_xlim([-0.1, 1.0])
        # El numero de clusters a insertar determinara el tamañp de cada barra
        # El coeficiente (n_clusters+1)*10 sera el espacio en blanco que dejaremos entre siluetas individuales de cada cluster para separarlas.
        axis1.set_ylim([0, len(X)+(k+1)*10])

        silhouette_avg = silhouette_score(X, labels)
        print(f'* Para k = {k} el promedio de la silueta es de: {silhouette_avg}')
        sample_silhouette_values = silhouette_samples(X, labels)

        y_lower = 10
        for i in range(k):
            # Agregamos la silueta del cluster i-ésimo
            ith_cluster_sv = sample_silhouette_values[labels == i]
            print(f'   - Para i = {i+1} la silueta del cluster value: {np.mean(ith_cluster_sv)}')
            # Ordenamos descendientemente las siluetas del cluester i-ésimo
            ith_cluster_sv.sort()

            # Calvulamos donde colocar la primera silueta en el eje vertical
            ith_cluster_size = ith_cluster_sv.shape[0]
            y_upper = y_lower + ith_cluster_size

            # Elegimos el color del cluster
            color = color_palette[i]

            axis1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sv, facecolor=color, alpha=0.7)

            # Etiquetamos dicho cluster con el numero en el centro
            axis1.text(-0.05, y_lower + 0.5 * ith_cluster_size, str(i+1))

            # Calculamos el nuevo y_lower para el siguiente cluster del grafico
            y_lower = y_upper + 10 # dejamos vacias 10 posiciones sin muestra
        
        axis1.set_title(f'Representacion de la silueta para k = {str(k)}')
        axis1.set_xlabel('S(i)')
        axis1.set_ylabel('ID del Cluster')
        # Fin de la representacion de la silueta

    # Plot de los k-means con los puntos respectivos
    plt.plot()
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.title(f'Clustering para k = {str(k)}')
    plt.scatter(x1, x2, c=label_color)
    plt.scatter(centers[0],
                centers[1],
                c=color_palette[k-1],
                marker='x')
    plt.show()

# Representacion del codo
plt.plot(K, ssw, 'bx-')
plt.xlabel('k')
plt.ylabel('SSw(k)')
plt.title('La tecnica del codo para encontrar el k optimo')
plt.show()

# Representacion del codo normalizado
plt.plot(K, 1-ssw/sst, 'bx-')
plt.xlabel('k')
plt.ylabel('1-norm(SSw(k))')
plt.title('La tecnica del codo normalizado para encontrar el k optimo')
plt.show()