import pandas as pd

data = pd.read_csv('./datasets/customer-churn-model/Customer Churn Model.txt')

print(data.head())

# Getting a dataset's column. Returns an object type Series
account_length = data['Account Length']
print(account_length.head())
print(type(account_length))

# Getting some dataset's columns. Returns an object type DatFrame
subset = data[['Account Length', 'Phone', 'Eve Charge', 'Day Calls']]
print(subset.head())
print(type(subset))

# Indicating desired columns
desired_columns = ['Account Length', 'Phone', 'Eve Charge', 'Night Calls']
subset = data[desired_columns]
print(subset.head())

# Indicating not desired columns
not_desired_columns = ['Account Length', 'Phone', 'Eve Charge', 'Night Calls', 'CustServ Calls', 'VMail Message', 'VMail Plan', 'Int\'l Plan']
all_columns_list = data.columns.values.tolist()
sublist = [x for x in all_columns_list if x not in not_desired_columns]
subset = data[sublist]
print(subset.head())