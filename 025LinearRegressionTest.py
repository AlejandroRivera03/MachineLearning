import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# y = a + b*x (linear regression model)
# x : 100 normal distributed values N(1.5, 2.5)     Media => 1.5    desviacion tipica => 2.5
# ye = 5 + 1.9*x + e
# e will be normally distributed N(0, 0.8)      Media => 0      desviacion tipica => 0.8

np.random.seed(19940903)

x = 1.5 + 2.5 * np.random.randn(100) # 100 normally distributed values for 'x'

res = 0 + 0.8 + np.random.randn(100)

y_pred = 5 + 1.9*x # test prediction (model)

y_actual = 5 + 1.9*x + res # test distributed 'y' values

x_list = x.tolist() # Changing array to a lists to create a dataframe
y_pred_list = y_pred.tolist()
y_actual_list = y_actual.tolist()
y_mean = [np.mean(y_actual) for i in range(1, len(x_list) + 1)]

data = pd.DataFrame(
    {
        'x': x_list,
        'y_actual': y_actual_list,
        'y_prediccion': y_pred_list,
        'y_mean': y_mean
    }
)

print(data.head())

plt.plot(data['x'], data['y_prediccion'])
plt.plot(data['x'], data['y_actual'], 'ro')
plt.plot(data['x'], y_mean, 'g')
plt.title('Valor Actual vs Predicción')
plt.show()

# SSD => Suma de los Cuadrados de las Diferencias
#       Distancia entre un punto disperso y la recta calculada
# SST => Suma de los Cuadrados Totales
#       Distancia entre el punto disperso y y_media
# SSR => Suma de los Cuadrados de la Regresion
#       Distancia entre el punto disperso y la recta calculada + distancia entr el puntp disperso y y_media

data['SSR'] = ( data['y_prediccion'] - np.mean(y_actual) )**2
data['SSD'] = ( data['y_prediccion'] - data['y_actual'] )**2
data['SST'] = ( data['y_actual'] - np.mean(y_actual) )**2

print(data.head())

print( f'SSR => {sum(data["SSR"])}' )
print( f'SSD => {sum(data["SSD"])}' )
print( f'SST => {sum(data["SST"])}' )
print( f'SSR+SSD = SST  => {sum(data["SSR"]) + sum(data["SSD"])}' )
print( f'R2 = SSR/SST => {sum(data["SSR"]) / sum(data["SST"])}' )