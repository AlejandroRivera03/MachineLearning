import pandas as pd
import os

mainpath = 'F:/Cursos/Machine Learning/python-ml-course/datasets'
filename = '/titanic/titanic3.csv'

# Pandas method to read a csv file
data = pd.read_csv(f'{mainpath}{filename}')

# property that returns the dataset columns in a list
print(data.columns.values)


# Reading a file with open() method
data = open( f'{mainpath}{filename}', 'r' )
cols = data.readline().strip().split(',') # strip() method without parameters remove left and right whitespaces, it can have a string parameter
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

infile = f'{mainpath}/customer-churn-model/Customer Churn Model.txt'
outfile = f'{mainpath}/customer-churn-model/My Customer Churn Model.txt'

with open(infile, 'r') as infiler:
    with open(outfile, 'w') as outfilew:
        for line in infiler:
            fields = line.strip().split(',')
            outfilew.write('\t'.join(fields))
            outfilew.write('\n')
df = pd.read_csv(outfile, sep='\t')

# print(df)