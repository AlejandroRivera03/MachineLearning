from scipy.spatial import distance_matrix
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def distance_matrix_to_dataframe(dd, col_name):
    return pd.DataFrame(dd, index=col_name, columns=col_name)

data = pd.read_csv('./datasets/movies/movies.csv', sep=';')

print(data)

movies = data.columns.values.tolist()[1:]

dd1 = distance_matrix(data[movies], data[movies], p=1) # Distancia de Manhattan
dd2 = distance_matrix(data[movies], data[movies], p=2) # Distancia euclidea
dd10 = distance_matrix(data[movies], data[movies], p=10)

print(distance_matrix_to_dataframe(dd1, data['user_id']))
# print(distance_matrix_to_dataframe(dd2, data['user_id']))
# print(distance_matrix_to_dataframe(dd10, data['user_id']))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=data['star_wars'], ys=data['lord_of_the_rings'], zs=data['harry_potter'])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()