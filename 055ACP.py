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
import plotly.graph_objects as ir
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./datasets/iris/iris.csv')

X = df.iloc[:,0:4].values
Y = df.iloc[:,4].values

# traces = []
# legend = {0:True, 1:True, 2:True, 3:True}

# colors = {'setosa': 'rgb(255,0,0)', 'versicolor': 'rgb(0,255,0)', 'virginica': 'rgb(0,0,255)'}

# for col in range(4):
#     for key in colors:
#         traces.append(ir.Histogram(x=X[Y==key, col], opacity=0.7, xaxis="x%s"%(col+1), marker=ir.Marker(color=colors[key]), name=key, showlegend=legend[col]))
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
        traces.append(ir.Histogram(x=X_std[Y==key, col], opacity=0.7, xaxis="x%s"%(col+1), marker=ir.Marker(color=colors[key]), name=key, showlegend=legend[col]))
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