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
new_point1 = [5,7] # Points to classify
new_point2 = [4,4]
new_point3 = [1,1]

[[plt.scatter(ii[0], ii[1], s=50, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_point1[0], new_point1[1], s=100, color='y')
plt.scatter(new_point2[0], new_point2[1], s=100, color='y')
plt.scatter(new_point3[0], new_point3[1], s=100, color='y')
plt.show()

def k_nearest_neighbors(data, predict, k=3):

    if len(data) >= k:
        warnings.warn('K es un valor menor que el número total de elementos a cvotar!!')
    
    distances = []
    for group in data:
        for feature in data[group]:
            # d = sqrt((feature[0]-predict[0])**2 + (feature[1]-predict[1])**2)
            # d = np.sqrt(np.sum((np.array(feature) - np.array(predict))**2))
            d = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([d, group])
    print(f'Distances => {distances}')
    votes = [i[1] for i in sorted(distances)[:k]] # sorted() ordena por la primera columna
    print(f'Votes => {votes}')
    vote_result = Counter(votes).most_common(1)
    print(f'Result => {vote_result}')
    return vote_result[0][0]

new_points = [new_point1, new_point2, new_point3]
for point in new_points:
    result = k_nearest_neighbors(dataset, point)
    plt.scatter(point[0], point[1], s=100, color=result)

[[plt.scatter(ii[0], ii[1], s=50, color=i) for ii in dataset[i]] for i in dataset]
plt.show()