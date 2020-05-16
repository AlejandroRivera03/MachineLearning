import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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