import pandas as pd

df = pd.read_csv('./datasets/gender-purchase/Gender Purchase.csv')

print(df.head())

print(f'\nshape => {df.shape}\n')

contingency_table = pd.crosstab(df['Gender'], df['Purchase'])
print(f'Tabla de contingencia => \n{contingency_table}')

print(f'\nSumatoria por filas (Gender) => \n{contingency_table.sum(axis=1)}')

print(f'\nSumatoria por columnas (Purchase) => \n{contingency_table.sum(axis=0)}')

print(f'\nProbabilidades => \n{contingency_table.astype("float").div(contingency_table.sum(axis=1), axis=0)}')

# CONDITIONAL PROBABILITY

print('\nProbabilidad de que un cliente compre sabiendo que es hombre')
print(f'P(Purchase | Male) = (Total purchases men / Total men) = (121 / 246) => {121/246}')

print('\nProbabilidad de que un cliente que compre un producto, sea mujer')
print(f'P(Female | Purchase) = (Total Purchases women / Total purchases) = (159 / 280) => {159/280}')

print(f'\nP(Purchase | Male) => {121/246}')
print(f'\nP(No purchase | Male) => {125/246}')
print(f'\nP(Purchase | Female) => {159/265}')
print(f'\nP(No purchase | Female) => {106/265}')