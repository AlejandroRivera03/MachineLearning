import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./datasets/Bank/bank.csv', sep=';')

print(data.head())
print(data.columns.values)

# changing the 'yes/no' y column values to zeros and ones
data["y"] = (data["y"]=="yes").astype(int)

print(data['education'].unique())

# Simplifying 'education' column
# Basic education generalized
data['education'] = np.where(data['education']=='basic.4y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='basic.6y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='basic.9y', 'Basic', data['education'])
# Better presentation for education values
data['education'] = np.where(data['education']=='high.school', 'High School', data['education'])
data['education'] = np.where(data['education']=='professional.course', 'Professional Course', data['education'])
data['education'] = np.where(data['education']=='university.degree', 'University Degree', data['education'])
data['education'] = np.where(data['education']=='illiterate', 'Illiterate', data['education'])
data['education'] = np.where(data['education']=='unknown', 'Unknown', data['education'])

# mean values respect y column
print(data.groupby('y').mean())

#mean values respect education column
print(data.groupby('education').mean())


## Personas con grado universitario tienden un poco mas a invertir
# Porcentaje respecto a cada categoria (nivel de educacion), el porcentaje es similar
pd.crosstab(data.education, data.y).plot(kind='bar')
plt.title('Frecuencia de compra en funcion del nivel de educacion')
plt.xlabel('Nivel de educacion')
plt.ylabel('Frecuencia de compra del producto')
plt.show()

table = pd.crosstab(data.education, data.y)
table.div( table.sum(1).astype(float), axis=0 ).plot(kind='bar', stacked=True)
plt.title('Diagrama apilado de educacion contra inversion')
plt.xlabel('Dia')
plt.ylabel('Proporcion de clientes')
plt.show()


## Porcentaje de cada categoria (estado civil) es similar
table = pd.crosstab(data.marital, data.y)
table.div( table.sum(1).astype(float), axis=0 ).plot(kind='bar', stacked=True)
plt.title('Diagrama apilado de estado civil contra inversion')
plt.xlabel('Estado civil')
plt.ylabel('Proporcion de clientes')
plt.show()


## Porcetaje de cada categoria (dia de la semana) es similar
pd.crosstab(data.day_of_week, data.y).plot(kind='bar')
plt.title('Frecuencia de compra en funcion del dia de la semana')
plt.xlabel('Dia de la semana')
plt.ylabel('Frecuencia de compra del producto')
plt.show()

table = pd.crosstab(data.day_of_week, data.y)
table.div( table.sum(1).astype(float), axis=0 ).plot(kind='bar', stacked=True)
plt.title('Diagrama apilado de dia contra inversion')
plt.xlabel('Dia')
plt.ylabel('Proporcion de clientes')
plt.show()


## En marzo y diciembre se ofrece menos invetir, pero en porcentaje hay mas caso de exito
pd.crosstab(data.month, data.y).plot(kind='bar')
plt.title('Frecuencia de compra en funcion del mes')
plt.xlabel('Mes')
plt.ylabel('Frecuencia de compra del producto')
plt.show()

table = pd.crosstab(data.month, data.y)
table.div( table.sum(1).astype(float), axis=0 ).plot(kind='bar', stacked=True)
plt.title('Diagrama apilado del mes contra inversion')
plt.xlabel('Mes')
plt.ylabel('Proporcion de clientes')
plt.show()


## Personas entre 25 y 60 años invierten en la bolsa de valores
data.age.hist()
plt.title('Histograma de la edad')
plt.xlabel('Edad')
plt.ylabel('Cliente')
plt.show()

pd.crosstab(data.age, data.y).plot(kind='bar')
plt.show()


## Quien pierde en invercion, raramente vuelve a invertir
# Quien no tiene conocimiento de la bolsa, raramente invierte
# Quien gana en inversion, normalmente vuelve a invertir
pd.crosstab(data.poutcome, data.y).plot(kind='bar')
plt.show()