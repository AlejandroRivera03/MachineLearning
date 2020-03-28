import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('./datasets/auto/auto-mpg.csv')
data['displacement'] = data['displacement'].fillna(data['displacement'].mean())
data['mpg'] = data['mpg'].fillna(data['mpg'].mean())

print(data.shape)

X = data['displacement']
X = X[:,np.newaxis]
Y = data['mpg']

lm = LinearRegression()
lm.fit(X, Y)

# with outliers
print(f'with outliers => {lm.score(X,Y)}') 
plt.plot(X, Y, 'ro')
plt.plot(X, lm.predict(X), color='blue')
# plt.show()

indexesToRemove = []
indexesToRemove[len(indexesToRemove):] = data[(data['displacement']>250)&(data['mpg']>35)].index.tolist()
indexesToRemove[len(indexesToRemove):] = data[(data['displacement']>300)&(data['mpg']>20)].index.tolist()
indexesToRemove[len(indexesToRemove):] = data[(data['displacement']<150)&(data['mpg']>35)].index.tolist()
print(f'{len(indexesToRemove)} rows removed to see improvement (10% aprox), but it must remove less percent')

# dropping some outliers by index
data_clean = data.drop(indexesToRemove) 

X = data_clean['displacement']
X = X[:,np.newaxis]
Y = data_clean['mpg']

lm = LinearRegression()
lm.fit(X, Y)

# without outliers
print(f'without outliers => {lm.score(X,Y)}') 
plt.plot(X, Y, 'go')
plt.title('Red ones (outliers) were remove to improve model')
# plt.plot(X, lm.predict(X), color='blue')
plt.show()