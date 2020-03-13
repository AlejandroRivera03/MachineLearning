import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def corr_coeff(df, var1, var2):
    df['corrn'] = (df[var1] - np.mean(df[var1])) * (df[var2] - np.mean(df[var2]))
    df['corr1'] = (df[var1] - np.mean(df[var1]))**2
    df['corr2'] = (df[var2] - np.mean(df[var2]))**2
    corr_p = sum(df['corrn']) / np.sqrt( sum(df['corr1']) * sum(df['corr2']) )
    return corr_p

print(f'''
    Correlation < 0  =>  Negative Correlation
    Correlation > 0  =>  Positive Correlation

    Correlation near 0  =>  Weak
    Correlation near -1 or 1  =>  Strong
''')

data_ads = pd.read_csv('./datasets/ads/Advertising.csv')
# print(data_ads.head())

cols = data_ads.columns.values

for x in cols:
    for y in cols:
        print(f'{x} - {y} => {str(corr_coeff(data_ads.copy(), x, y))}')

plt.plot(data_ads['TV'], data_ads['Sales'], 'ro')
plt.title('Gasto en TV vs Ventas del Producto')
plt.show()
plt.plot(data_ads['Radio'], data_ads['Sales'], 'go')
plt.title('Gasto en Radio vs Ventas del Producto')
plt.show()
plt.plot(data_ads['Newspaper'], data_ads['Sales'], 'bo')
plt.title('Gasto en Newspaper vs Ventas del Producto')
plt.show()


# corr() => pandas method that returns a matrix with correlation between columns values
print(data_ads.corr())

plt.matshow(data_ads.corr())
plt.show()