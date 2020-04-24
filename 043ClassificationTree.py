import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

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