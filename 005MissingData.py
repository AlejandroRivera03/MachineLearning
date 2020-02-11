import pandas as pd

mainpath = 'F:/Cursos/Machine Learning/python-ml-course/datasets'
filename = '/titanic/titanic3.csv'
fullpath = f'{mainpath}{filename}'

data = pd.read_csv(fullpath)

# pandas method that says if each column values is null or not
print(pd.isnull(data['body']))

# pandas method that says if each columns values is not null
print(pd.notnull(data['body']))

print('Valores nulos en columna body => {}'.format(pd.isnull(data['body']).values.ravel().sum()))
print('Valores no nulos en columna body => {}'.format(pd.notnull(data['body']).values.ravel().sum()))

# MISSING DATA, OPTIONS...

# Deleting null data
# dropna() method  =>  dataset.dropna(axis=, how='',)

# axis: 
#       0 => indicate rows
#       1 => indicate columns
# how: 
#       all => deletes the complete row/column that all its vales are null
#       any => deletes the complete row/column that has at least a null value



# General filling
# fillna() method => dataset.fillna(parameter)

# dataset.fillna(parameter):
#       parameter(num/str) => fill the whole null values with the given parameter value
# dataset['COLUMN1'].fillna(parameter)
#       parameter(num/str) => fill the null values in COLUMN1 with the given parameter value

# This method doesn't overwrite the original dataset, so...
# dataset = dataset.fillna(value) => whole nulls in dataset
# dataset['ColumnX'] = dataset['ColumnX'].fillna(value) => nulls in ColumnX



# Filling nulls with the not null values' average
# dataset['age'] = dataset['age'].fillna( data['age'].mean() )



# Filling null with the last / next known value

# method 'ffill' overwrite the null value with the first 'forward' known value
# dataset['age'] = dataset['age'].fillna(method='ffill')
# method 'backfill' overwrite the null value with the first 'bacjward' known value
# dataset['age'] = dataset['age'].fillna(method='backfill')