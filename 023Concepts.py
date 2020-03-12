import pandas as pd
import numpy as np
import random as rd
import scipy.stats as sp
import matplotlib.pyplot as plt

# Setting a seed to repeat random numbers
rd.seed(0)
# 60 number from a range of 100000
random_numbers = rd.sample(range(100000), 60) 
# print(random_numbers)



# MEDIDAS DE CENTRALIZACION
# NOS SIRVE PARA VER COMO SE SITUAN LOS DATOS

# Media (promedio)
print(f'Media => {np.mean(random_numbers)}')

# Mediana
print(f'Mediana => {np.median(random_numbers)}')

# Moda
print(f'Moda => {sp.mode(random_numbers)}')

# Percentil
print(f'Precentil 25 => {np.percentile(random_numbers, 25)}') # 1er Cuartil 
print(f'Precentil 50 => {np.percentile(random_numbers, 50)}') # 2do Cuartil
print(f'Precentil 75 => {np.percentile(random_numbers, 75)}') # 3er Cuartil



# MEDIDAS DE DISPERSION

# Varianza
# Indica que tanto se alejan los valores respecto de la media (esta elevada al cuadrado)
print(f'Varianza => {np.var(random_numbers)}')

# Desviacion tipica
# Indica que tanto se alejan los valores respecto de la media (raiz cuadrada de la varianza)
print(f'Desviacion tipica => {np.std(random_numbers)}')

# Coeficiente de variacion
# nos mide la variabilidad relativa entre la desviación típica entre la media
print(f'desviacion tipica / media * 100 => {np.std(random_numbers)/np.mean(random_numbers)*100}')



# MEDIDAS DE ASIMETRIA

# Asimetría de Fisher
#     • Si el coeficiente es = 0; Significa que vuestra función es perfectamente
#       simetríca, se distribuye igual, por ejemplo la distribución normal. Raro es
#       que salga cero
#     • Si el coeficiente es >0; Significa que cuánto más positivo es este valor más
#       desplazada está la distribución hacía la izquierda, de modo que tenemos una
#       asimetría positiva, nos queda la media muy por encima de la distribución.
#     • Si el el coeficiente es <0; Significa que cuánto más negativo es este
#       valor más desplazado está la distribución hacía la derecha, de modo que
#       tenemos una asimetría negativa, nos queda la media muy por debajo de la
#       distribución.
print(f'Asimetria Fisher => {sp.skew(random_numbers)}')

# Curtosis
#     • = 0 Mesocúrtica Distribución perfecta, asemejada a la distribución normal en
#       forma, no en valores. Está compensado tanto el centro como las colas.
#     • > 0 Leptocúrtica Distribución donde se le concentran mucho los datos en el
#       valor central, y apenas tiene cola.
#     • < 0 Platicúrtica Distribución donde hay pocos valores que se concentren
#       respecto al valor central (media) y hay muchos que aparecen hacia las colas,
#       se concentran más en los laterales. Existe valor central, pero también hay
#       mucha presencia de colas directamente en la distribución de nuestros datos.
print(sp.kurtosis(random_numbers))

plt.hist(random_numbers)
plt.show()