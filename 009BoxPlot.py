import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/customer-churn-model/Customer Churn Model.txt')

plt.boxplot(data['Day Calls'])
plt.ylabel('Numero de llamadas diarias')
plt.title('Boxplot de las llamadas diarias')

print(data['Day Calls'].describe())
# Rango intercuartilico
IQR = data['Day Calls'].quantile(0.75) - data['Day Calls'].quantile(0.25)
print(f'Interquartile Range => {IQR}')

# Bigote inferior
lower = data['Day Calls'].quantile(0.25) - 1.5*IQR
print(lower)

# Bigote superior
higher = data['Day Calls'].quantile(0.75) + 1.5*IQR
print(higher)

plt.show()