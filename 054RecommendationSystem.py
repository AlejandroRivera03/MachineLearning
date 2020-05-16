import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./datasets/ml-100k/u.data.csv', sep='\t', header=None)

print(f'Dataset shape => {df.shape}')

df.columns = ['UserID', 'ItemID', 'Rating', 'TimeStamp']

print(df.head())

# Hist of movies by rating, score
print(f'Total rates => {df.groupby(["Rating"])["UserID"].count()}')
plt.hist(df.Rating)
plt.show()

plt.hist(df.TimeStamp)
plt.show()

# Most rated movies
plt.hist(df.groupby(['ItemID'])['ItemID'].count())
plt.show()