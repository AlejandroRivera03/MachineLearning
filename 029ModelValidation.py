import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

data = pd.read_csv('./datasets/ads/Advertising.csv')

np.random.seed(19940903)

# Normal distributed data. Same length than dataset
a = np.random.randn(len(data))

check = (a<0.8) # Variable that help us to separate data aproximately at 80% to trainning and 20% to test
training = data[check] # data's subset to train model
testing = data[~check] # data's subset to test model

lm = smf.ols(formula='Sales~TV+Radio', data=training).fit()
print(lm.summary())
# Sales = 2.5388 + 0.0468*TV + 0.1943*Radio

# Testing model with data from the dataset which were not considered in the model
sales_pred = lm.predict(testing)
print(sales_pred)

SSD = sum((testing['Sales']-sales_pred)**2)
RSE = np.sqrt(SSD/(len(testing)-2-1)) # 2 because TV and Radio were considered to the model
sales_mean = np.mean(testing['Sales'])
error = RSE/sales_mean
print(f'SSD => {SSD}')
print(f'RSE => {RSE}')
print(f'error => {error}')