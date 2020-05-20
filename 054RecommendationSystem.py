import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('./datasets/ml-100k/u.data.csv', sep='\t', header=None)

print(f'Dataset shape => {df.shape}')

df.columns = ['UserID', 'ItemID', 'Rating', 'TimeStamp']

print(df.head())

# Hist of movies by rating, score
print(f'Total rates => {df.groupby(["Rating"])["UserID"].count()}')
plt.hist(df.Rating)
plt.show()

plt.hist(df.TimeStamp)
plt.show()

# Most rated movies
plt.hist(df.groupby(['ItemID'])['ItemID'].count())
plt.show()

n_users = df.UserID.unique().shape[0]
n_items = df.ItemID.unique().shape[0]

print(f'Numero total de usuarios => {n_users}\nNumero total de peliculas => {n_items}')

# Creating a matrix to contrast users and users' movies rated
ratings = np.zeros((n_users, n_items))

for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]

# print(ratings)

sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0]*ratings.shape[1])
sparsity *= 100
print('Coeficiente de sparseidad: {:4.2f}%'.format(sparsity))

# Creating training and validation sets

ratings_train, ratings_test = train_test_split(ratings, test_size=0.3, random_state=42)
print(f'ratings_train.shape => {ratings_train.shape}\nratings_test.shape => {ratings_test.shape}')

# Filtro colaborativo basado en Usuarios
# 1 - Matriz de similaridad entre los usuarios (distancia del coseno)
# 2 - Predecir la valoracion desconocida de un item 'i' para un usuario activo 'u' basandonos en la suma ponderada de todas las valoraciones del resto de usuarios para dicho item
# 3 - Recomendaremos los nuevos items a los usuarios segun lo establecido en los pasos anteriores

# usando la ditancia del coseno, nos devolvera que los 0 son los mas parecidos y los 1 los menos 
# parecidos, revertiremos esto a los 0 los menos parecidos y los 1 los mas parecidos
# 1
sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(ratings_train)
print(f'sim_matrix.shape => {sim_matrix.shape}') # train users versus themselves

# print(sim_matrix)

users_predictions = sim_matrix.dot(ratings_train) / np.array([np.abs(sim_matrix).sum(axis=1)]).T
# print(users_predictions)

# Error cuadratico medio
def get_mse(preds, actuals):
    preds = preds[actuals.nonzero()].flatten()
    actuals = actuals[actuals.nonzero()].flatten()
    return mean_squared_error(preds, actuals)

print(get_mse(users_predictions, ratings_train))

print(get_mse(users_predictions, ratings_test))

# Filtro colaborativo basado en KNN

k = 10
neighbors = NearestNeighbors(k, 'cosine')
neighbors.fit(ratings_train)

top_k_distances, top_k_users = neighbors.kneighbors(ratings_train, return_distance=True)

print(f'top_k_distances.shape => {top_k_distances.shape}')
print(f'top_k_distances[0] => {top_k_distances[0]}')
print(f'top_k_users.shape => {top_k_users.shape}')
print(f'top_k_users[0] => {top_k_users[0]}')

users_predicts_k = np.zeros(ratings_train.shape)
for i in range(ratings_train.shape[0]): # Para cada usuario del conjunto de entrenamiento
    users_predicts_k[i,:] = top_k_distances[i].T.dot(ratings_train[top_k_users][i]) / np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T

print(f'users_predicts_k.shape => {users_predicts_k.shape}')
print(f'users_predicts_k => {users_predicts_k}')

print(get_mse(users_predicts_k, ratings_train))

# Filtrado colaborativo basado en Items

n_movies = ratings_train.shape[1]

neighbors = NearestNeighbors(n_movies, 'cosine')
neighbors.fit(ratings_train.T)

top_k_distances, top_k_items = neighbors.kneighbors(ratings_train.T, return_distance=True)
print(f'top_k_distances.shape => {top_k_distances.shape}')
print(f'top_k_items.shape => {top_k_items.shape}')

print(top_k_items)

item_preds = ratings_train.dot(top_k_distances) / np.array([np.abs(top_k_distances).sum(axis=1)])

print(get_mse(item_preds, ratings_train))
print(get_mse(item_preds, ratings_test))

# Filtrado colaborativo basado en KNN

k = 30
neighbors = NearestNeighbors(k, 'cosine')
neighbors.fit(ratings_train.T)
top_k_distances, top_k_items = neighbors.kneighbors(ratings_train.T, return_distance=True)

# print(top_k_distances[0]) Example first movie and its most similar movies (ids)
# print(top_k_distances[0]) Example first movie and its most similar movies (distances)

preds = np.zeros(ratings_train.T.shape)
for i in range(ratings.T.shape[0]):
    preds[i, :] = top_k_distances[i].dot(ratings_train.T[top_k_items][i]) / np.array([np.abs(top_k_distances[i]).sum(axis=0)]).T

print(f'Error cuadrado medio de ratings_train => {get_mse(preds, ratings_train.T)}')
print(f'Error cuadrado medio de ratings_test => {get_mse(preds, ratings_test.T)}')