import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors

df = pd.read_csv('./datasets/cancer/breast-cancer-wisconsin.data.txt', header=None)

print(f'Dataset =>\n{df.head()}')

df.columns = ['name', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'class'] # Setting names to columns

df = df.drop(['name'], 1) # Removing this unnecessary column
Y = df['class']
X = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']]
print(f'Predictor Variables =>\n{X.head()}')
print(f'Target =>\n{Y.head()}')