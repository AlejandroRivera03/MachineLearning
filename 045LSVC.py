# Linear support vector classifier

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn import svm

X = [1, 5, 1.5, 8, 1, 9]
Y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(X,Y)
plt.show()

data = np.array(list(zip(X,Y)))
print(data)

target = [0, 1, 0, 1, 0, 1]
print(f'We classify data by index (group 0 and group 1) => {target}')

classifier = svm.SVC(kernel='linear', C=1.0)
classifier.fit(data, target)

point_to_predict = np.array([0.57, 0.67]).reshape(1,2) # It needs this format
print(f'Prediction of {point_to_predict} is => {classifier.predict(point_to_predict)}')

point_to_predict = np.array([10.32, 12.67]).reshape(1,2) # It needs this format
print(f'Prediction of {point_to_predict} is => {classifier.predict(point_to_predict)}')