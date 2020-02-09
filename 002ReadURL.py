import pandas as pd
import urllib3
import csv

url = 'http://winterolympicsmedals.com/medals.csv'

medals_data = pd.read_csv(url)

# print(medals_data)

# Code to get data from internet
http = urllib3.PoolManager()
r = http.request('GET', url)
response = r.data
print(r.status)
# print(response)

# Method to decode encoded data
str_data = response.decode('utf-8')

lines = str_data.split('\n')

col_names = lines[0].split(',')
n_cols = len(col_names)

counter = 0
main_dict = {}
# Adding the keys to the dict
for col in col_names:
    main_dict[col] = []

# Adding each data to its corresponding in the dict
for line in lines:
    if(counter > 0):
        values = line.split(',')
        for i in range(len(col_names)):
            main_dict[col_names[i]].append(values[i])
    counter += 1

print(f'El data set tiene {counter} filas y {n_cols} columnas')

# Finally convert the dictionaty to a data frame
medals_df = pd.DataFrame(main_dict)

mainpath = 'F:/Cursos/Machine Learning/python-ml-course/datasets'
filename = '/athletes/my_downloaded_medals.'
fullpath = mainpath+filename

medals_df.to_csv(fullpath+'csv')
medals_df.to_json(fullpath+'json')
medals_df.to_excel(fullpath+'xlsx')