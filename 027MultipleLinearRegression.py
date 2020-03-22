import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

data = pd.read_csv('./datasets/ads/Advertising.csv')

# Model with TV and Newspaper values
lm2 = smf.ols(formula='Sales~TV+Newspaper', data=data).fit()
print(f'Parametros => \n{lm2.params}\n')
print('Modelo => Sales = 5.774948 + 0.046901*TV + 0.044219*Newspaper\n')
print(f'P-Valores => \n{lm2.pvalues}\n') # pvalues entre mas pequeños, mejor
print(f'R Cuadrada => {lm2.rsquared}\n') # Entre mas alto mejor (es porcentaje)
print(f'R Cuadrada ajustada => {lm2.rsquared_adj}\n')
sales_pred = lm2.predict(data[['TV', 'Newspaper']])
# print(sales_pred)

SSD = sum((data['Sales']-sales_pred)**2)
print(f'SSD => {SSD}') # Suma de los cuadrados de las diferencias

RSE = np.sqrt(SSD/(len(data)-3)) # 3 porque son dos variables (TV y Newspaper) menos 1
print(f'RSE => {RSE}')

sales_m = np.mean(data['Sales'])
error = RSE / sales_m
print(f'error => {error}') # Porcentaje de datos que el modelo no puede explicar

# print(lm2.summary())

# Model with TV and Radio values
lm3 = smf.ols(formula='Sales~TV+Radio', data=data).fit()
print(f'Parametros => \n{lm3.params}\n')
print('Modelo => Sales = 2.921100 + 0.045755*TV + 0.187994*Radio\n')
print(f'P-Valores => \n{lm3.pvalues}\n') # pvalues entre mas pequeños, mejor
print(f'R Cuadrada => {lm3.rsquared}\n') # Entre mas alto mejor (es porcentaje)
print(f'R Cuadrada ajustada => {lm3.rsquared_adj}\n')
sales_pred = lm3.predict(data[['TV', 'Radio']])
# print(sales_pred)

SSD = sum((data['Sales']-sales_pred)**2)
print(f'SSD => {SSD}') # Suma de los cuadrados de las diferencias

RSE = np.sqrt(SSD/(len(data)-3)) # 3 porque son dos variables (TV y Radio) menos 1
print(f'RSE => {RSE}')

sales_m = np.mean(data['Sales'])
error = RSE / sales_m
print(f'error => {error}') # Porcentaje de datos que el modelo no puede explicar

# print(lm3.summary())

# Comparacion entre modelo de TV y Newspaper con TV y Radio
# Mejor modelo => TV y Radio, Razones
# P-Valor desminuye (Prob(F-statistic))
# R-Squared y R-Squared adj Incrementan
# Estadistico mas grande (F-statistic)
# Coeficiente de radio es alto

# Model with TV Radio and Newspaper values
lm4 = smf.ols(formula='Sales~TV+Radio+Newspaper', data=data).fit()
print(f'Parametros => \n{lm4.params}\n')
print('Modelo => Sales = 2.938889 + 0.045765*TV + 0.188530*Radio + (-0.001037)*Newspaper\n')
print(f'P-Valores => \n{lm4.pvalues}\n') # pvalues entre mas pequeños, mejor
print(f'R Cuadrada => {lm4.rsquared}\n') # Entre mas alto mejor (es porcentaje)
print(f'R Cuadrada ajustada => {lm4.rsquared_adj}\n')
sales_pred = lm4.predict(data[['TV', 'Radio', 'Newspaper']])
# print(sales_pred)

SSD = sum((data['Sales']-sales_pred)**2)
print(f'SSD => {SSD}') # Suma de los cuadrados de las diferencias

RSE = np.sqrt(SSD/(len(data)-4)) # 4 porque son dos variables (TV Radio y Newspaper) menos 1
print(f'RSE => {RSE}')

sales_m = np.mean(data['Sales'])
error = RSE / sales_m
print(f'error => {error}') # Porcentaje de datos que el modelo no puede explicar

# print(lm4.summary())