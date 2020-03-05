import pandas as pd

def zeroPad(number):
    zeros = str(number)
    if( len( zeros ) < 3 ):
        zeros = f'0{zeros}'
        zeros = zeroPad(zeros)
    return zeros

filepath = './datasets/distributed-data/'

data = pd.read_csv(f'{filepath}001.csv')

for i in range(2,333):
    filename = zeroPad(i)
    File = f'{filepath}{filename}.csv'
    temp_data = pd.read_csv(File)
    data = pd.concat([data, temp_data], axis=0)

print(data.shape)