import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from graphviz import Source
import graphviz as gp
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import os
os.environ["PATH"] += os.pathsep + 'F:/Archivos de programa/Graphviz2.38/bin'

data = pd.read_csv('./datasets/boston/Boston.csv')
print(f'shape => {data.shape}')
print(data.head())

colnames = data.columns.values.tolist()
predictors = colnames[:13]
target = colnames[13]
X = data[predictors]
Y = data[target]

regtree = DecisionTreeRegressor(min_samples_split=30, min_samples_leaf=10, max_depth=5, random_state=1)
regtree.fit(X,Y)

preds = regtree.predict(data[predictors])

data['preds'] = preds

# print(data[['preds', 'medv']])

with open('resources/boston_rtree.dot', 'w') as dotfile:
    export_graphviz(regtree, out_file=dotfile, feature_names=predictors)
    dotfile.close()

file = open('resources/boston_rtree.dot', 'r')
text = file.read()

Source(text, filename='resources/boston_rtree').view()

cv = KFold(shuffle=True, random_state=1)
scores = cross_val_score(regtree, X, Y, scoring='neg_mean_squared_error', cv=cv, n_jobs=1)
score = np.mean(scores)
print(scores)
print(score) # error

print(list(zip(predictors, regtree.feature_importances_)))

# RANDOM FORESTS

forest = RandomForestRegressor(n_jobs=2, oob_score=True, n_estimators=500) # n_estimators => number of trees
forest.fit(X,Y)

data['rforest_pred'] = forest.oob_prediction_
# print(data[['r_forest_pred', 'medv']])

data['rforest_error2'] = (data['rforest_pred'] - data['medv'])**2
print(sum(data['rforest_error2'])/len(data)) # Mean error

print(forest.oob_score_) # Similar to R^2 (Coeficiente de determinacion)