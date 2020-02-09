import pandas as pd

mainpath = 'F:/Cursos/Machine Learning/python-ml-course/datasets'
filename = '/titanic/titanic3.xls'

# Pandas method to read a excel file
titanic3 = pd.read_excel(f'{mainpath}{filename}', 'titanic3')

titanic3.to_csv(f'{mainpath}/titanic/mytitaniccustom.csv')
titanic3.to_excel(f'{mainpath}/titanic/mytitaniccustom.xls')
titanic3.to_json(f'{mainpath}/titanic/mytitaniccustom.json')