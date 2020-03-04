from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./datasets/customer-churn-model/Customer Churn Model.txt')



# Dividing using normal distribution
a = np.random.randn(len(data))
check = ( a < (0.8) )
training = data[check]
testing = data[~check]
# print(training)
# print(testing)



# Dividing using sklearn library
train, test = train_test_split(data, test_size=0.2)
print(len(train))
print(len(test))



# Dividing using shuffle method (from sklearn library)
data2 = sklearn.utils.shuffle(data)
cut_id = int(0.75*len(data2))
train_data = data2[:cut_id]
test_data = data2[cut_id:]