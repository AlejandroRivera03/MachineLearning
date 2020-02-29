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

# Simple grouping by gender (2 groups)
grouped_gender = data.groupby('Gender')

print(grouped_gender.groups)

for names, groups in grouped_gender:
    print(names)
    print(groups)

# Getting a simple group
print(f'Female group =>\n{grouped_gender.get_group("Female")}')

# Double grouping by gender and economic status (6 groups )
double_group = data.groupby(['Gender', 'Economic Status'])

for names, groups in double_group:
    print(names)
    # print(groups)

# Getting a double grouping group
print(double_group.get_group(('Male', 'Rich')))

# Data grouping operations
# data sum
print(double_group.sum())

# data mean (media-promedio)
print(double_group.mean())

# data size
print(double_group.size())

print(double_group.describe())

grouped_income = double_group['Income']
print(grouped_income.describe())

print(double_group.aggregate(
    {
        'Income': np.sum, # Suma
        'Age': np.mean, # Media
        'Height': np.std, # Desviacion estandar
    }
))

print(double_group.aggregate(
    {
        'Age': np.mean,
        'Height': lambda h:(np.mean(h))/np.std(h) # tipificacion de la edad
    }
))

print(double_group.aggregate([np.sum, np.mean, np.std])) # function(s) to apply to all dataset columns

print(double_group.aggregate([lambda x:np.mean(x) / np.std(x)]))