import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

data_auto = pd.read_csv('./datasets/auto/auto-mpg.csv')

print(data_auto.shape)

data_auto['mpg'] = data_auto['mpg'].dropna()
data_auto['horsepower'] = data_auto['horsepower'].dropna()

plt.plot(data_auto['horsepower'], data_auto['mpg'], 'ro')
plt.xlabel('Caballos de fuerza')
plt.ylabel('Consumo (Millas por galon)')
plt.title('CV vs MPG')
plt.show()


# MODELO LINEAL
# mpg = a + b*horsepower

X = data_auto['horsepower'].fillna(data_auto['horsepower'].mean())
Y = data_auto['mpg'].fillna(data_auto['mpg'].mean())
X_data = X[:,np.newaxis]

print(type(X), type(Y), type(X_data))

lm = LinearRegression()
lm.fit(X_data, Y)

plt.plot(X, Y, 'ro')
plt.plot(X, lm.predict(X_data), color='blue')
plt.show()

SSD = np.sum((Y - lm.predict(X_data))**2)
RSE = np.sqrt(SSD/(len(X_data)-1))
y_mean = np.mean(Y)
error = RSE/y_mean

print('\nModelo Lineal')
print(f'R^2 => {lm.score(X_data, Y)}')
print(f'SSD => {SSD}')
print(f'RSE => {RSE} (+/-)')
print(f'y_mean => {y_mean}')
print(f'error => {error}')

# MODELO DE REGRESION CUADRATICO
# mpg = a + b*horsepower^2

X_data = X**2
X_data = X_data[:,np.newaxis]

lm = LinearRegression()
lm.fit(X_data, Y)

SSD = np.sum((Y - lm.predict(X_data))**2)
RSE = np.sqrt(SSD/(len(X_data)-1))
y_mean = np.mean(Y)
error = RSE/y_mean

print('\nModelo Cuadratico')
print(f'R^2 => {lm.score(X_data, Y)}')
print(f'SSD => {SSD}')
print(f'RSE => {RSE} (+/-)')
print(f'y_mean => {y_mean}')
print(f'error => {error}')

# MODELO DE REGRESION LINEAL Y CUADRATICO
# mpg = a + b*horsepower + c*horsepower^2

poly = PolynomialFeatures(degree=2)
X_data = poly.fit_transform(X[:,np.newaxis])
lm = linear_model.LinearRegression()
lm.fit(X_data, Y)

print(f'\nConstante => {lm.intercept_}, Variables => {lm.coef_}')
print('mpg = 55.02619244708123 + (-0.43404318)*horsepower + 0.00112615*horsepower^2')

SSD = np.sum((Y - lm.predict(X_data))**2)
RSE = np.sqrt(SSD/(len(X_data)-1))
y_mean = np.mean(Y)
error = RSE/y_mean

print('\nModelo Lineal y Cuadratico')
print(f'R^2 => {lm.score(X_data, Y)}')
print(f'SSD => {SSD}')
print(f'RSE => {RSE} (+/-)')
print(f'y_mean => {y_mean}')
print(f'error => {error}')