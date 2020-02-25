import numpy as np
import pandas as pd
import random

data = pd.read_csv('./datasets/customer-churn-model/Customer Churn Model.txt')

print('random number between 1 and 100')
print(f'numpy.random.randint(1,100) => {np.random.randint(1,100)}\n')

print('Most common way to get a randon number')
print(f'numpy.random.random() => {np.random.random()}\n')

def randint_list(n, a, b):
    x = []
    for i in range(n):
        x.append(np.random.randint(a,b))
    return x

print(randint_list(25, 1, 50))

print('Random numbers between 1 and 100, and multiplo of 7(plus 1. due to lower limit)')
print(f'random.randrange(1, 100, 7) => {random.randrange(1, 100, 7)}')


# numpy Shuffle 

print('\nnumpy shuffle (Mix values)\n')
a = list(range(15))
print(f'a = list(range(15)) => {a}')
np.random.shuffle(a)
print(f'np.random.shuffle(a) => {a}\n')

b = np.arange(15)
print(f'b = np.arange(15) => {b}')
np.random.shuffle(b)
print(f'np.random.shuffle(b) => {b}\n')

column_list = data.columns.values.tolist()
print('column_list = data.columns.values.tolist()')
print(f'column_list => {column_list}')
# Getting a random value from column list with choice method
print(f'np.random.choice(column_list) => {np.random.choice(column_list)}')

# Seed
# Stablishing a 'Seed' is very important when we want to reproduce
# an experiment, due to the seed allows us to repeat the random values
np.random.seed(2018)
for i in range(5):
    print(np.random.random())