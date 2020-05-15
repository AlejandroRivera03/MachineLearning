import warnings
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import style
from collections import Counter

dataset = {
    'k': [[1,2], [2,3], [3,1]],
    'r': [[6,5], [7,7], [8,6]]
}
new_point = [5,7]

[[plt.scatter(ii[0], ii[1], s=50, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_point[0], new_point[1], s=100)
plt.show()

def k_nearest_neighbors(data, precit, k=3):

    if len(data) >= k:
        warnings.warn('K es un valor menor que el número total de elementos a cvotar!!')
    