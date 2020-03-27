# LINEAR REGRESION WITH scikit_learn PACKAGE AND CATEGORICAL VARIABLES

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./datasets/ecom-expense/Ecom Expense.csv')

# print(df.head()) # 'Gender' and 'City Tier' are the categorical variables

dummy_gender = pd.get_dummies(df['Gender'], prefix='Gender')
dummy_city_tier = pd.get_dummies(df['City Tier'], prefix='City')

# print(dummy_gender.head())
# print(dummy_city_tier.head())

column_names = df.columns.values.tolist()
df_new = df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()
df_new = df_new[column_names].join(dummy_city_tier)
print(df_new.head())

feature_cols = ['Monthly Income', 'Transaction Time', 'Gender_Female', 'Gender_Male', 'City_Tier 1', 'City_Tier 2', 'City_Tier 3', 'Record']

X = df_new[feature_cols]
Y = df_new['Total Spend']

lm = LinearRegression()
lm.fit(X,Y)

print(f'Constante => {lm.intercept_}')
print(f'Variables => {lm.coef_}')

print(f'\n{list(zip(feature_cols, lm.coef_))}') # Predictor Variables Before Simplifying

print(f'\nR^2 => {lm.score(X,Y)}')

# Total_Spend = -79.41713030136634 + 'Monthly Income'*0.14753898049205724 + 'Transaction Time'*0.15494612549589348 + 'Gender_Female'*-131.0250132555455 + 'Gender_Male'*131.02501325554587 + 'City_Tier 1'*76.76432601049547 + 'City_Tier 2'*55.138974309232296 + 'City_Tier 3'*-131.90330031972775 + 'Record'*772.2334457445639

df_new['prediction'] = -79.41713030136634 + df_new['Monthly Income']*0.14753898049205724 + df_new['Transaction Time']*0.15494612549589348 + df_new['Gender_Female']*-131.0250132555455 + df_new['Gender_Male']*131.02501325554587 + df_new['City_Tier 1']*76.76432601049547 + df_new['City_Tier 2']*55.138974309232296 + df_new['City_Tier 3']*-131.90330031972775 + df_new['Record']*772.2334457445639

print(df_new.head())

SSD = np.sum((df_new['prediction'] - df_new['Total Spend'])**2)
RSE = np.sqrt( SSD / (len(df_new)-len(feature_cols)-1) )
sales_mean = np.mean(df_new['Total Spend'])
error = RSE/sales_mean

print(f'SSD => {SSD}')
print(f'RSE => {RSE}')
print(f'sales_mean => {sales_mean}')
print(f'error => {error*100}%')

# Eliminando variables dummy redundantes

dummy_gender = pd.get_dummies(df['Gender'], prefix='Gender').iloc[:,1:] # Only the Gender_male column
dummy_city_tier = pd.get_dummies(df['City Tier'], prefix='City').iloc[:,1:] # The City_Tier 2 and City_Tier 3 columns
print(dummy_gender.head())
print(dummy_city_tier.head())

column_names = df.columns.values.tolist()
df_new = df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()
df_new = df_new[column_names].join(dummy_city_tier)
print(df_new.head())

feature_cols = ['Monthly Income', 'Transaction Time', 'Gender_Male', 'City_Tier 2', 'City_Tier 3', 'Record']
X = df_new[feature_cols]
Y = df_new['Total Spend']
lm = LinearRegression()
lm.fit(X,Y)

print(f'Constante => {lm.intercept_}')
print(list(zip(feature_cols, lm.coef_))) # Predictor Variables After Simplifying
print(f'R^2 => {lm.score(X,Y)}')