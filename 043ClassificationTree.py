import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Source
import graphviz as gp
from sklearn.tree import DecisionTreeClassifier, export_graphviz
os.environ["PATH"] += os.pathsep + 'F:/Archivos de programa/Graphviz2.38/bin'

data = pd.read_csv('./datasets/iris/iris.csv')

print(f'shape => {data.shape}')
print(data.head())

plt.hist(data.Species)
plt.show()

colnames = data.columns.values.tolist()
predictors = colnames[:4]
target = colnames[4]

data['is_train'] = np.random.uniform(0, 1, len(data))<=0.75
# print(data.head())

train, test = data[data['is_train'] == True], data[data['is_train'] == False]

tree = DecisionTreeClassifier(criterion='entropy', min_samples_split=20, random_state=99)
tree.fit(train[predictors], train[target])

preds = tree.predict(test[predictors])
print(pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions']))

with open('resources/iris_dtree.dot', 'w') as dotfile:
    export_graphviz(tree, out_file=dotfile, feature_names=predictors)
    dotfile.close()

file = open('resources/iris_dtree.dot', 'r')
text = file.read()

Source(text, filename='resources/iris_dtree').view()