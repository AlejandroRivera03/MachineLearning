import numpy as np
import pandas as pd
# import plotly.plotly as py
import chart_studio.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go

N = 2000
random_x = np.random.randn(N)
random_y = np.random.randn(N)

# SCATTER PLOT SENCILLO
trace = go.Scatter(x=random_x, y=random_y, mode='markers')
fig = go.Figure(data=trace)
fig.show()

# plot_url = py.plot([trace], filename='basic-scatter-online', auto_open=True)
# print(plot_url)

# GRAFICOS CONBINADOS
N = 200
rand_x = np.linspace(0, 1, N)
rand_y0 = np.random.randn(N) + 3
rand_y1 = np.random.randn(N)
rand_y2 = np.random.randn(N) - 3

trace0 = go.Scatter(x=rand_x, y=rand_y0, mode='markers', name='Puntos')
trace1 = go.Scatter(x=rand_x, y=rand_y1, mode='lines', name='Lineas')
trace2 = go.Scatter(x=rand_x, y=rand_y2, mode='lines+markers', name='Puntos y lineas')
data = [trace0, trace1, trace2]

fig = go.Figure(data=data)
fig.show()

# ESTILIZADO DE GRAFICOS
trace = go.Scatter(x=random_x, y=random_y, name='Puntos de estilo', mode='markers',
                    marker=dict(size=12, color='rgba(140,20,20,0.8)', line=dict(width=2, color='rgb(10,10,10)')))
layout = dict(title='Scatter Plot Estilizado', xaxis=dict(zeroline=False), yaxis=dict(zeroline=False))
fig = go.Figure(dict(data=[trace], layout=layout))
fig.show()

trace = go.Scatter(x=random_x, y=random_y, name='Puntos de estilo', mode='markers',
                    marker=dict(size=8, color='rgba(10,80,220,0.25)', line=dict(width=1, color='rgb(10,10,80)')))
fig = go.Figure(dict(data=[trace], layout=layout))
fig.show()

trace = go.Histogram(x=random_x)
layout = dict(title='Histogram', xaxis=dict(zeroline=False), yaxis=dict(zeroline=False))
fig = go.Figure(dict(data=[trace], layout=layout))
fig.show()

trace = go.Box(x=random_x, fillcolor='rgba(180,25,95,0.6)')
layout = dict(title='Box', xaxis=dict(zeroline=False), yaxis=dict(zeroline=False))
fig = go.Figure(dict(data=[trace], layout=layout))
fig.show()

# INFORMACION AL HACER HOVER
data = pd.read_csv('./datasets/usa-population/usa_states_population.csv')

print(data.head())

N = 53
c = ['hsl('+str(h)+', 50%, 50%)' for h in np.linspace(0, 360, N)]

l = []
y = []
for i in range(int(N)):
    y.append((2000+i))
    trace0 = go.Scatter(
        x = data['Rank'],
        y = data['Population'] + i*1000000,
        mode = 'markers',
        marker = dict(size=14, line=dict(width=1), color=c[i], opacity=0.3),
        # name = data['State']
    )
    l.append(trace0)

layout = go.Layout(
    title = 'Poblacion de los estados de USA',
    hovermode = 'closest',
    xaxis = dict(title='ID', ticklen=5, zeroline=False, gridwidth=2),
    yaxis = dict(title='Poblacion', ticklen=5, gridwidth=2),
    showlegend = False
)

fig = go.Figure(data = l, layout = layout)
fig.show()

trace = go.Scatter(
    y = np.random.randn(1000),
    mode = 'markers',
    marker = dict(size=16, color=np.random.randn(1000), colorscale='Viridis', showscale=True)
)
fig = go.Figure(data = trace)
fig.show()

# DATASETS MUY GRANDES
N = 1000000
trace = go.Scattergl(
    x = np.random.randn(N),
    y = np.random.randn(N),
    mode = 'markers',
    marker = dict(color='#BAD5FF', line=dict(width=1))
)
fig = go.Figure(data = trace)
fig.show()