import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 1000000

data = pd.DataFrame(
    {
        'A': np.random.randn(n), # Normal distribution (standard)
        'B': 1.5 + 2.5 * np.random.randn(n), # Normal distribution with media = 1.5 and desviacion tipica = 2.5
        'C': np.random.uniform(5, 32, n), # Uniform distribution between 5 and 32
    }
)

# MEAN => Media
# STD => Desviacion tipica
print(data.describe())

plt.hist(data['A']) # Normal distribution standard
plt.show()

plt.hist(data['B']) # Normal distribution with media = 1.5 and desviacion tipica = 2.5
plt.show()

plt.hist(data['C']) #  Uniform distribution between 5 and 32
plt.show()


data = pd.read_csv('./datasets/customer-churn-model/Customer Churn Model.txt')
print(data.head())
column_names = data.columns.values.tolist()
a = len(column_names)
new_data = pd.DataFrame(
    {
        'Column Name': column_names,
        'A': np.random.randn(a),
        'B': np.random.uniform(0, 1, a),
    }, index=range(42, 42+a)
)
print(new_data)