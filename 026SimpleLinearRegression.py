# Linear regression using statsmodel package
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

data = pd.read_csv('./datasets/ads/Advertising.csv')

lm = smf.ols(formula='Sales~TV', data=data).fit()
print(f'alpha and beta => \n{lm.params}')
# Modelo Lineal =>  Sales = 7.032594 + 0.047537*TV

print(f'p-values => \n{lm.pvalues}')

print(f'R^2 => {lm.rsquared}')

print(f'R^2 ajustada => {lm.rsquared_adj}')

print(lm.summary())

sales_pred = lm.predict(pd.DataFrame(data['TV']))

data.plot(kind='scatter', x='TV', y='Sales')
plt.plot(pd.DataFrame(data['TV']), sales_pred, c='red', linewidth=2)
plt.show()

data['sales_pred'] = 7.032594 + 0.047537*data['TV']
data['RSE'] = (data['Sales']-data['sales_pred'])**2

SSD = sum(data['RSE'])
print(f'SSD => {SSD}')

RSE = np.sqrt(SSD/(len(data)-2))
print(f'RSE => {RSE}') # Varianza, rango +/-

sales_m = np.mean(data['Sales'])
print(f'Sales Mean => {sales_m}') # promedio de ventas

error = RSE/sales_m
print(f'Error percent => {error}') # Porcentaje de datos que el modelo no puede explicar

plt.hist(data['Sales']-data['sales_pred'])
plt.show()