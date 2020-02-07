import pandas as pd
import urllib3
import csv

url = 'http://winterolympicsmedals.com/medals.csv'

medals_data = pd.read_csv(url)

# print(medals_data)

http = urllib3.PoolManager()
r = http.request('GET', url)
response = r.data
print(r.status)
# print(response)