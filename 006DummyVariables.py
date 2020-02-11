import pandas as pd

mainpath = 'F:/Cursos/Machine Learning/python-ml-course/datasets'
filename = '/titanic/titanic3.csv'
fullpath = f'{mainpath}{filename}'

data = pd.read_csv(fullpath)

# Separating the sex column in two columns
dummy_sex = pd.get_dummies(data['sex'], prefix='sex')
print(data['sex'].head(6))
print(dummy_sex.head(6))

column_name = data.columns.values.tolist()
print(column_name)

# Removing sex column, axis=1 means column
data = data.drop(['sex'], axis=1)

# Adding the dummy variables
data = pd.concat([data, dummy_sex], axis=1)

print(data.head(6))

# Function to create dummys
def createDummies(data_frame, column_name):
    dummy = pd.getdummies(data_frame[column_name], prefix=column_name)
    data_frame = data_frame.drop(column_name, axis=1)
    data_frame = pd.concat([data_frame, dummy], axis=1)
    return data_frame