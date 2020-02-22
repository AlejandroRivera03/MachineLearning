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
# sublist = [x for x in all_columns_list if x not in not_desired_columns]
sublist = list(set(all_columns_list)-set(not_desired_columns))
subset = data[sublist]
print(subset.head())

# print specific row by indexes
print(data[100:115])

# sub set by conditional
# rows where 'Day Mins' more than 300
data1 = data[data['Day Mins']>300]
print(f'rows where \'Day Mins\' more than 300 =>  {data1.shape}')

# rows where 'State' equal than NY (New York)
data2 = data[data['State']=='NY']
print(f'rows where \'State\' equal than NY (New York) => {data2.shape}')

# rows where 'Day Mins' more than 300 and 'State' equal than NY
data3 = data[(data['Day Mins']>300) & (data['State']=='NY')]
print(f'rows where \'Day Mins\' more than 300 and \'State\' equal than NY => {data3.shape}')

# rows where 'Day Mins' more than 300 or 'State' equal than NY
data4 = data[(data['Day Mins']>300) | (data['State']=='NY')]
print(f'rows where \'Day Mins\' more than 300 or \'State\' equal than NY => {data4.shape}')

# rows where 'Night Calls' are more that 'Day Calls'
data5 = data[(data['Day Calls']<data['Night Calls'])]
print(f'rows where \'Night Calls\' are more that \'Day Calls\' => {data5.shape}')