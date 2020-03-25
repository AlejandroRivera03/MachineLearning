from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

data = pd.read_csv('./datasets/ads/Advertising.csv')

feature_cols = ['TV', 'Radio', 'Newspaper'] # variables predictoras disponibles

X = data[feature_cols]
Y = data['Sales']

estimator = SVR(kernel='linear')
selector = RFE(estimator, 2, step=1).fit(X,Y) # 2 means => 2 variables predictoras a usar

print(selector.support_) # Ambos datos nos dicen cuales son las mejores variables predictoras
print(selector.ranking_) # Less value means more importance

X_pred = X[['TV', 'Radio']]
lm = LinearRegression()
lm.fit(X_pred, Y)

print(f'Constante => {lm.intercept_}')
print(f'Variables => {lm.coef_}')
print(f'R^2 => {lm.score(X_pred, Y)}')