import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from graphviz import Source
import graphviz as gp
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
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

# VISUALIZACION DEL ARBOL DE DECISION

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

# CROSS VALIDATION PARA LA PODA

X = data[predictors]
Y = data[target]

for i in range(1,11):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=i, min_samples_split=20, random_state=99)
    tree.fit(X,Y)
    cv = KFold(n_splits=10, shuffle=True, random_state=1) # n=X.shape[0], 
    scores = cross_val_score(tree, X, Y, scoring='accuracy', cv=cv, n_jobs=1)
    score = np.mean(scores)
    print(f'Score para i = {i} => {score}\n\t{tree.feature_importances_}')

forest = RandomForestClassifier(n_jobs=2, oob_score=True, n_estimators=100)
forest.fit(X,Y)

# print(forest.oob_decision_function_) # Classification

print(forest.oob_score_) # Precision del modelo