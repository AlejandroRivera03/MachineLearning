import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv('./datasets/wine/winequality-red.csv', sep=';')

print(df.head())

print(f'\nshape => {df.shape}\n')

plt.hist(df['quality'])
plt.title('Calidad de vinos')
plt.xlabel('Calidad')
plt.ylabel('Cantidad de vinos')
plt.show()

print(f'\nAgrupando por calidad y calculando medias =>\n{df.groupby("quality").mean()}\n')

# Normalizando data
df_norm = (df-df.min())/(df.max()-df.min())
print(f'Dataset normalizado =>\n{df_norm.head()}')

# Crating hierarchy clusters and making an histogram
clus = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(df_norm)
md_h = pd.Series(clus.labels_) # Relation each wine with a cluster by index
plt.hist(md_h)
plt.title('Histograma de clusters por jerarquia')
plt.xlabel('Cluster')
plt.ylabel('Numero de vinos en el cluster')
plt.show()

print(clus.children_)

# Dendrogram
Z = linkage(df_norm, 'ward')
plt.figure(figsize=(19,9))
plt.title('Dendrograma de los vinos cluteriados')
plt.ylabel('Distancia')
dendrogram(Z, leaf_rotation=90.0, leaf_font_size=8.0, truncate_mode='lastp', p=12, show_leaf_counts=True, show_contracted=True, color_threshold=4)
plt.show()

# Crating K-means clusters and making an histogram
model = KMeans(n_clusters=6)
model.fit(df_norm)
print(model.labels_) # Relation each wine with a cluster by index

md_k = pd.Series(model.labels_) # Getting the cluster each wine belong to

df_norm['clust_h'] = md_h
df_norm['clust_k'] = md_k
print(df_norm.head())
plt.hist(md_k)
plt.title('Histograma de clusters por k-means')
plt.xlabel('Cluster')
plt.ylabel('Numero de vinos en el cluster')
plt.show()
print(f'\nCentroides de cada cluster en 12 dimensiones =>\n{model.cluster_centers_}')

# FINAL INTERPRETATION (hierarchy)
print(f'\nInterpretacion final por jerarquia (agrupando por cluster y calculando medias) =>\n{df_norm.groupby("clust_h").mean()}')
# FINAL INTERPRETATION (K-means)
print(f'\nInterpretacion final por k-means (agrupando por cluster y calculando medias) =>\n{df_norm.groupby("clust_k").mean()}')