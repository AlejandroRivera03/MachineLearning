import pandas as pd

mainpath = 'F:/Cursos/Machine Learning/python-ml-course/datasets'
filename = '/titanic/titanic3.csv'
fullpath = f'{mainpath}{filename}'

data = pd.read_csv(fullpath)
print('Imprimiendo los primeros 10 =>')
print(data.head(10)) # By default it returns first 5
print('imprimiento los ultimos 10 =>')
print(data.tail(10)) # By default ir returns last 5
print('Dimensiones (filas, columnas) =>')
print(data.shape) # It returns the data set dimensions (rows and columns)
print('Columnas del dataset =>')
print(data.columns.values) # it returns the columns' headers
print('Estadisticas generales =>')
print(data.describe()) # It returns basic and general dataset's statistics to know a gereral dataset review
print('Tipos de datos =>')
print(data.dtypes) # It return the data type for each column