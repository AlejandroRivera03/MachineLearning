import numpy as np
from scipy.cluster.vq import vq, kmeans

np.random.seed(19940903)
data = np.random.random(90).reshape(30,3)
print(data)

c1 = np.random.choice(range(len(data)))
c2 = np.random.choice(range(len(data)))

clust_centers = np.vstack([data[c1], data[c2]])
print(f'\ncenters =>\n{clust_centers}')

# 1st array => it means which element (by position) belong to each cluster
# 2nd array => it means the distance between the element to the cluster that belongs
print(f'\n1st array: belongs to. 2nd array: distance =>\n{vq(data, clust_centers)}')

# kmeans with pre-defined centers
print(f'\nkmeans with pre-fefined centers =>\n{kmeans(data, clust_centers)}')

# kmeans with certain amount of centers that we want
print(f'\nkmeans with certain amount of centers that we want (2) => \n{kmeans(data, 2)}')
