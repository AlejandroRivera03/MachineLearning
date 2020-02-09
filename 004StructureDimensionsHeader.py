import pandas as pd

mainpath = 'F:/Cursos/Machine Learning/python-ml-course/datasets'
filename = '/titanic/titanic3.csv'
fullpath = f'{mainpath}{filename}'

data = pd.read_csv(fullpath)
print('Imprimiendo los primeros 10 =>')
print(data.head(10))
print('imprimiento los ultimos 10 =>')
print(data.tail(10))
print('Dimensiones (filas, columnas) =>')
print(data.shape)
print('Columnas del dataset =>')
print(data.columns.values)
print('Estadisticas generales =>')
print(data.describe())
print('Tipos de datos =>')
print(data.dtypes)