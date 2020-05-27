import pandas as pd
# import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as g_objs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sk_pca

df = pd.read_csv('./datasets/iris/iris.csv')

X = df.iloc[:,0:4].values # getting the predictor variables
y = df.iloc[:,4].values # getting the result
X_std = StandardScaler().fit_transform(X)
# print(f'X_std =>\n{X_std}')

acp = sk_pca(n_components=2) # the '2' was calculated in the last python file
Y = acp.fit_transform(X_std)

# print(f'Y =>\n{Y}')

results = []

for name in ('setosa', 'versicolor', 'virginica'):
    result = g_objs.Scatter(x=Y[y==name, 0], y=Y[y==name, 1], mode='markers', name=name,
                                marker=g_objs.scatter.Marker(size=8, line=g_objs.Line(color='rgba(255,255,255,0.2)', width=0.5), opacity=0.75))
    results.append(result)

data = g_objs.Data(results)
layout = g_objs.Layout(xaxis=g_objs.layout.XAxis(title='CP1', showline=False), yaxis=g_objs.layout.YAxis(title='CP2', showline=False))

fig = g_objs.Figure(data=data, layout=layout)
fig.show()