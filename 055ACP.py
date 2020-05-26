# Analisis de Componentes Principales

# 1 - Estandarizar los datos (para cada una de las 'm' observaciones)
# 2 - Obtener los vectores y valores propios a partir de la matriz de covarianzas o de correlaciones o incluso la tecnica de singular vector decomposition
# 3 - Obtener los valores propios en orden descendente y quedarnos con los 'p' que se correspondan a los 'p' mayores y asi distribuir el numero de variables del dataset (p<m)
# 4 - Contruir la matriz de proyeccion 'W' a partir de los 'p' vectores propios
# 5 - Tranformar el dataset original 'X' a traves de 'W' para asi obtener datos en el subespacio dimensional de dimension 'p', que sera 'Y'

import numpy as np
import pandas as pd
import plotly as py
import plotly.tools as tls
import matplotlib.pyplot as plt
import plotly.graph_objects as ir
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./datasets/iris/iris.csv')

X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

# traces = []
# legend = {0:True, 1:True, 2:True, 3:True}

# colors = {'setosa': 'rgb(255,0,0)', 'versicolor': 'rgb(0,255,0)', 'virginica': 'rgb(0,0,255)'}

# for col in range(4):
#     for key in colors:
#         traces.append(ir.Histogram(x=X[y==key, col], opacity=0.7, xaxis="x%s"%(col+1), marker=ir.Marker(color=colors[key]), name=key, showlegend=legend[col]))
#     legend = {0:False, 1:False, 2:False, 3:False}

# data = ir.Data(traces)
# layout = ir.Layout(barmode='overlay',
#                 xaxis=ir.layout.XAxis(domain=[0, 0.25], title='Long. Sepalos (cm)'),
#                 xaxis2=ir.layout.XAxis(domain=[0.3, 0.5], title='Anch. Sepalos (cm)'),
#                 xaxis3=ir.layout.XAxis(domain=[0.55, 0.75], title='Long. Petalos (cm)'),
#                 xaxis4=ir.layout.XAxis(domain=[0.8, 1], title='Anch. Petalos (cm)'),
#                 yaxis=ir.layout.YAxis(title='Numero de ejemplares'),
#                 title='Distribucion de los rasgos de las diferentes flores iris')

# fig = ir.Figure(data=data, layout=layout)
# fig.show()

# Point 1 (Estandarizar data)
X_std = StandardScaler().fit_transform(X)

traces = []
legend = {0:True, 1:True, 2:True, 3:True}

colors = {'setosa': 'rgb(255,0,0)', 'versicolor': 'rgb(0,255,0)', 'virginica': 'rgb(0,0,255)'}

for col in range(4):
    for key in colors:
        traces.append(ir.Histogram(x=X_std[y==key, col], opacity=0.7, xaxis="x%s"%(col+1), marker=ir.Marker(color=colors[key]), name=key, showlegend=legend[col]))
    legend = {0:False, 1:False, 2:False, 3:False}

data = ir.Data(traces)
layout = ir.Layout(barmode='overlay',
                xaxis=ir.layout.XAxis(domain=[0, 0.25], title='Long. Sepalos (cm)'),
                xaxis2=ir.layout.XAxis(domain=[0.3, 0.5], title='Anch. Sepalos (cm)'),
                xaxis3=ir.layout.XAxis(domain=[0.55, 0.75], title='Long. Petalos (cm)'),
                xaxis4=ir.layout.XAxis(domain=[0.8, 1], title='Anch. Petalos (cm)'),
                yaxis=ir.layout.YAxis(title='Numero de ejemplares'),
                title='Distribucion de los rasgos de las diferentes flores iris')

fig = ir.Figure(data=data, layout=layout)
fig.show()

# Point 2 (Obtener los vectores y valores propios 'matrix de covarianza')
# Vector de las medias
mean_vect = np.mean(X_std, axis=0)
print(f'Vector de las medias \n{mean_vect}')

# Matriz de covarianzas
cov_matrix = (X_std - mean_vect).T.dot((X_std - mean_vect))/(X_std.shape[0]-1)
print(f'La matriz de covarianzas es \n{cov_matrix}') # 1's en diagonal (aprox)

np.cov(X_std.T)

# Vectores y valores propios (con libreria)
eig_vals, eig_vectors = np.linalg.eig(cov_matrix)
print(f'Valores propios \n{eig_vals}')
print(f'Vectores propios \n{eig_vectors}')

# Point 2 (Obtener los vectores y valores propios 'matriz de correlaciones')
# Matriz de correlaciones
corr_matrix = np.corrcoef(X_std.T)
print(f'La matriz de correlaciones \n{corr_matrix}')

eig_vals_corr, eig_vectors_corr = np.linalg.eig(corr_matrix)
print(f'Valores propios \n{eig_vals_corr}')
print(f'Vectores propios \n{eig_vectors_corr}')

# Point 2 (Obtener los vectores y valores propios 'singular value decomposition')
u,s,v = np.linalg.svd(X_std.T)
print(f'u =>\n{u}')
print(f's =>\n{s}')
print(f'v =>\n{v}')

# Point 3
for ev in eig_vectors:
    print(f'La longitud del VP es: {np.linalg.norm(ev)}')

eigen_pairs = [(np.abs(eig_vals[i]), eig_vectors[:,i]) for i in range(len(eig_vals))]
eigen_pairs.sort()
eigen_pairs.reverse()
print(f'eigen_pairs =>\n{eigen_pairs}')

print('Valores propios en orden descendente:')
for ep in eigen_pairs:
    print(ep[0])

total_sum = sum(eig_vals)
var_exp = [(i/total_sum)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print(f'var_exp => {var_exp}') # Porcentaje de informacion valiosa que cada columna aporta al dataset
print(f'cum_var_exp => {cum_var_exp}') # suma acumulada

plot1 = ir.Bar(x=['CP %s'%i for i in range(1,5)], y=var_exp, showlegend=False)
plot2 = ir.Scatter(x=['CP %s'%i for i in range(1,5)], y=cum_var_exp, showlegend=True, name="% de Varianza Explicada Acumulada")

data = ir.Data([plot1, plot2])
layout = ir.Layout(xaxis=ir.XAxis(title='Componentes principales'), yaxis=ir.YAxis(title='Porcentaje de varianza explicada'), title='Porcentaje de variabilidad explicada por cada componente principal')

fig = ir.Figure(data=data, layout=layout)
fig.show()

# Point 4
W = np.hstack((eigen_pairs[0][1].reshape(4,1), eigen_pairs[1][1].reshape(4,1))) # Getting first 2 columns (first represents 74% and the second one 22% (96% aprox))
print(f'W => {W}')

# Point 5
Y = X_std.dot(W)
print(f'Y =>\n{Y}')

results = []

for name in ('setosa', 'versicolor', 'virginica'):
    result = ir.Scatter(x=Y[y==name, 0], y=Y[y==name, 1], mode='markers', name=name, marker=ir.Marker(size=12, line=ir.Line(color='rgba(220, 220, 220, 0.15)', width=0.5), opacity=0.8))
    results.append(result)

data = ir.Data(results)
layout = ir.Layout(showlegend=True, scene=ir.layout.Scene(xaxis=ir.XAxis(title='Componente Principal 1'), yaxis=ir.YAxis(title='Componente Principal 2')))

fig = ir.Figure(data=data, layout=layout)
fig.show()