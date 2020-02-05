import pandas as pd
import os

mainpath = 'F:/Cursos/Machine Learning/python-ml-course/datasets'
filename = '/titanic/titanic3.csv'

data = pd.read_csv(f'{mainpath}{filename}')

print(data.columns.values)


# Reading a file with open() method
data = open( f'{mainpath}{filename}', 'r' )
cols = data.readline().strip().split(',')
n_cols = len(cols)
counter = 0
main_dict = {}

for col in cols:
    main_dict[col] = []
print(main_dict)

for line in data:
    values = line.strip().split(',')
    for i in range(len(cols)):
        main_dict[cols[i]].append(values[i])
    counter += 1
print(f'El data set tiene {counter} filas y {n_cols} columnas')
# work the content data to after that convert it to a DatFrame
df = pd.DataFrame(main_dict)
# print(df)