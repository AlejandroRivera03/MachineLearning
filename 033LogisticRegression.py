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


# RATIO DE PROBABILIDADES (Cociente de probabilidades)

# Cociente = (0, infinito)
# Cociente < 1  =>  Mas probable el fracaso que el exito
# Cociente = 1  =>  Equiprobable (misma probabilidad de que pase a que no pase)
# Cociente > 1  =>  Mas probable el exito que el fracaso (puede tender a infinito)

# pm = probabilidad de compra sabiendo que es hombre
# pf = probabilidad de compra sabiendo que es mujer

# cocientem   =   pm / (1 - pm)   =   pm / ~pm 
# cocientef   =   pf / (1 - pf)   =   pf / ~pf

pm = 121 / 246
pf = 159 / 265
cocientem = pm/(1-pm)
cocientef = pf/(1-pf)
print(f'male probability ratio => {cocientem}')
print(f'female probability ratio => {cocientef}')

print(121/125)
print(159/106)