import pandas as pd
import numpy as np

# Categorical variables
gender = ['Female', 'Male']
income = ['Poor', 'Middle Class', 'Rich']

n = 500
gender_data = []
income_data = []

for i in range(n):
    # Generating a dataset with random values from the categorical variables
    gender_data.append(np.random.choice(gender))
    income_data.append(np.random.choice(income))

height = 160 + 30 * np.random.randn(n)
weight = 65 + 25 * np.random.randn(n)
age = 30 + 12 * np.random.randn(n)
income = 18000 + 8000 * np.random.rand(n)

data = pd.DataFrame(
    {
        'Gender': gender_data,
        'Economic Status': income_data,
        'Height': height,
        'Weight': weight,
        'Age': age,
        'Income': income,
    }
)

print(data.head(10))