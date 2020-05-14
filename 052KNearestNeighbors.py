import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors

df = pd.read_csv('./datasets/cancer/breast-cancer-wisconsin.data.txt', header=None)

print(f'Dataset =>\n{df.head()}')

df.columns = ['name', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'class'] # Setting names to columns

df.replace('?', -99999, inplace=True) # there are '?' in the dataset
Y = df['class']
X = df[['name', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']]
print(f'Predictor Variables =>\n{X.head()}')
print(f'Target =>\n{Y.head()}')

# Clasificacion de los K Vecinos
#   Sin limpieza
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(f'Accuracy without data cleaning => {accuracy}')


#   Con limpieza
df = df.drop(['name'], 1) # Removing this unnecessary column
X = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']] # The difference is here, skip 'name' column
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(f'Accuracy with data cleaning => {accuracy}')

# Classifying new data
sample_measure1 = np.array([4,2,1,1,1,2,3,2,1]).reshape(1,-1)
sample_measure2 = np.array([5,10,10,3,7,3,8,10,2]).reshape(1,-1)
sample_measure3 = np.array([2,2,4,4,2,2,6,2,4]).reshape(1,-1)

predict = clf.predict(sample_measure1)
print(f'sample_measure1 => {predict} ({"Maligno" if predict[0] == 4 else "No Maligno"}). data => {sample_measure1}')
predict = clf.predict(sample_measure2)
print(f'sample_measure2 => {predict} ({"Maligno" if predict[0] == 4 else "No Maligno"}). data => {sample_measure2}')
predict = clf.predict(sample_measure3)
print(f'sample_measure3 => {predict} ({"Maligno" if predict[0] == 4 else "No Maligno"}). data => {sample_measure3}')
# 2 = No maligno
# 4 = Maligno