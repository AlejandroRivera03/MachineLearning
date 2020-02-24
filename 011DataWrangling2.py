import pandas as pd

data = pd.read_csv('./datasets/customer-churn-model/Customer Churn Model.txt')

# Getting 3 columns and the first 50 rows
subset_first_50 = data[['Day Mins', 'Night Mins', 'Account Length']][:50]
print(subset_first_50.shape)

# print(data.ix[1:10, 3:6]) ix method deprecated
print('\nFirst 10 rows from 3rd to 6th columns')
print(data.iloc[1:10, 3:6])
print('\nAll rows from 3rd to 6th columns')
print(data.iloc[:, 3:6])
print('\nFirst 10 rows from all columns')
print(data.iloc[1:10, :])
print('\nFirst 10 rows from 1nd, 5th and 7th columns')
print(data.iloc[1:10, [2,5,7]])
print('\n# 1st, 5th, 8th and 36th rows from 2nd, 5th and 7th columns')
print(data.iloc[[1,5,8,36], [2,5,7]])
print('\n# 1st, 5th, 8th and 36th rows from Area Code, VMail Plan and Day Mins columns')
print(data.loc[[1,5,8,36], ['Area Code', 'VMail Plan', 'Day Mins']])

print('\nCreating a new column (Total Mins)')
data['Total Mins'] = data['Day Mins'] + data['Night Mins'] + data['Eve Mins']
print(data.loc[[1,2,3,4,5], ['Day Mins', 'Night Mins', 'Eve Mins', 'Total Mins']])

print('\nCreating a new column (Total Calls)')
data['Total Calls'] = data['Day Calls'] + data['Night Calls'] + data['Eve Calls']
print(data.loc[[1,2,3,4,5], ['Day Calls', 'Night Calls', 'Eve Calls', 'Total Calls']])