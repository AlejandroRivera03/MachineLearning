# Propagacion de la afinidad

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt

centers = [[1,1], [-1,-1], [1,-1]]
X, labels = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)

plt.scatter(X[:,0], X[:,1], c=labels, s=20, cmap='autumn')
plt.show()

def report_affinity_propagation(X):
    af = AffinityPropagation(preference=-50).fit(X)
    cluster_center_ids = af.cluster_centers_indices_
    clust_labels = af.labels_
    n_clust = len(cluster_center_ids)

    print(f'Numero estimado de clusters => {n_clust}')
    print(f'Homogeneidad => {metrics.homogeneity_score(labels, clust_labels)}')
    print(f'Completitud => {metrics.completeness_score(labels, clust_labels)}')
    print(f'V-measure => {metrics.v_measure_score(labels, clust_labels)}')
    print(f'R2 Ajustado => {metrics.adjusted_rand_score(labels, clust_labels)}')
    print(f'Informacion mutua ajustada => {metrics.adjusted_mutual_info_score(labels, clust_labels)}')
    print(f'Coeficiente de la silueta => {metrics.silhouette_score(X, labels, metric="sqeuclidean")}')

    plt.figure(figsize=(16,9))
    plt.clf()

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for k, col in zip(range(n_clust), colors):
        class_members = (clust_labels == k)
        clust_center = X[cluster_center_ids[k]]
        plt.plot(X[class_members, 0], X[class_members, 1], col+'.')
        plt.plot(clust_center[0], clust_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            plt.plot([clust_center[0], x[0]], [clust_center[1], x[1]], col)
    
    plt.title(f'Numero estimado de clusters: {n_clust}')
    plt.show()

report_affinity_propagation(X)