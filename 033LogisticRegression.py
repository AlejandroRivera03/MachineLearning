import pandas as pd

df = pd.read_csv('./datasets/gender-purchase/Gender Purchase.csv')

print(df.head())

print(f'\nshape => {df.shape}\n')

contingency_table = pd.crosstab(df['Gender'], df['Purchase'])
print(f'Tabla de contingencia => \n{contingency_table}')

print(f'\nSumatoria por filas (Gender) => \n{contingency_table.sum(axis=1)}')

print(f'\nSumatoria por columnas (Purchase) => \n{contingency_table.sum(axis=0)}')

print(f'\nProbabilidades => \n{contingency_table.astype("float").div(contingency_table.sum(axis=1), axis=0)}')