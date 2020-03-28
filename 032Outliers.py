import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('./datasets/auto/auto-mpg.csv')

print(data.shape)

X = data['displacement'].fillna(data['displacement'].mean())
X = X[:,np.newaxis]
Y = data['mpg'].fillna(data['mpg'].mean())

lm = LinearRegression()
lm.fit(X, Y)

# with outliers
print(f'with outliers => {lm.score(X,Y)}') 
plt.plot(X, Y, 'ro')
plt.plot(X, lm.predict(X), color='blue')
# plt.show()


# print(data[(data['displacement']>250)&(data['mpg']>35)])
# print(data[(data['displacement']>300)&(data['mpg']>20)])
# print(data[(data['displacement']<150)&(data['mpg']>40)])

# dropping some outliers by index
# indexes were taken from the result of the before prints
data_clean = data.drop([395, 258, 305, 372, 251, 316, 329, 331, 332, 333, 336, 337, 402]) 

X = data_clean['displacement'].fillna(data_clean['displacement'].mean())
X = X[:,np.newaxis]
Y = data_clean['mpg'].fillna(data_clean['mpg'].mean())

lm = LinearRegression()
lm.fit(X, Y)

# without outliers
print(f'without outliers => {lm.score(X,Y)}') 
plt.plot(X, Y, 'go')
# plt.plot(X, lm.predict(X), color='blue')
plt.show()
