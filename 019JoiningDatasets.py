import pandas as pd

red_wine = pd.read_csv('./datasets/wine/winequality-red.csv', sep=';')
print(red_wine.shape)
print(red_wine.columns.values)

white_wine = pd.read_csv('./datasets/wine/winequality-white.csv', sep=';')
print(white_wine.shape)
print(white_wine.columns.values)

wine_data = pd.concat([red_wine, white_wine], axis=0)
print(wine_data.shape)



data1 = wine_data.head(10)
data2 = wine_data[300:310]
data3 = wine_data.tail(10)

wine_scramble = pd.concat([data2, data1, data3], axis=0)
print(wine_scramble)