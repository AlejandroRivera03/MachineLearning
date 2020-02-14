import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/customer-churn-model/Customer Churn Model.txt')

# Regla de sturges para distribuir un histograma
k = int(np.ceil(1+np.log2(3333))) # 3333 => Total rows

plt.hist(data['Day Calls'], bins=k) # bins => segments
plt.xlabel('Numero de llamadas al dia')
plt.ylabel('Frecuencia')

plt.show()