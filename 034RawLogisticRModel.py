import numpy as np
from numpy import linalg
import statsmodels.api as sm

# Funcion de entorno L(b)
# L(beta) = sumatoria desde i=1 hasta n de Pi^yi * (1 - Pi)^yi
def likelihood(y, pi):
    total_sum = 1
    sum_in = list(range(1, len(y)+1))
    for i in range(len(y)):
        sum_in[i] = np.where(y[i] == 1, pi[i], 1-pi[i])
        total_sum = total_sum * sum_in[i]
    return total_sum

# Calcular probabilidades para cada observacion
# Pi = P(xi) = 1 / (1 + e^(-sumatoria desde j=0 hasta k de betaj * xij))
def logitprobs(X, beta):
    n_rows = np.shape(X)[0]
    n_cols = np.shape(X)[1]
    pi = list(range(1, n_rows+1))
    expon = list(range(1, n_rows+1))
    for i in range(n_rows):
        expon[i] = 0
        for j in range(n_cols):
            ex = X[i][j] * beta[j]
            expon[i] = ex + expon[i]
        with np.errstate(divide='ignore', invalid='ignore'):
            pi[i] = 1 / (1 + np.exp(-expon[i]))
    return pi

# Calcular la matriz diagonal W
# W = diag(Pi (1 - Pi)) desde i=1 hasta n
def findW(pi):
    n = len(pi)
    W = np.zeros(n*n).reshape(n,n)
    for i in range(n):
        # print(i)
        W[i,i] = pi[i]*(1 - pi[i])
        W[i,i].astype(float)
    return W

# Ordenar la solucion de la funcion logistica
# B sub(n+1) = Bn - (f(Bn) / f'(Bn))
# f(B) = X(Y - P)}
# f'(B) = X * W * X^t
def logistics(X, Y, limit):
    nrow = np.shape(X)[0]
    bias = np.ones(nrow).reshape(nrow, 1)
    X_new = np.append(X, bias, axis=1)
    ncol = np.shape(X_new)[1]
    beta = np.zeros(ncol).reshape(ncol, 1)
    root_dif = np.array(range(1, ncol+1)).reshape(ncol, 1)
    iter_i = 10000
    while(iter_i > limit):
        # print(f'Iter: i {str(iter_i)}, limit: {str(limit)}')
        pi = logitprobs(X_new, beta)
        # print(f'Pi: {str(pi)}')
        W = findW(pi)
        # print(f'W: {str(W)}')
        num = (np.transpose(np.matrix(X_new)) * np.matrix(Y - np.transpose(pi)).transpose())
        den = (np.matrix(np.transpose(X_new)) * np.matrix(W) * np.matrix(X_new))
        root_dif = np.array(linalg.inv(den) * num)
        beta = beta + root_dif
        # print(f'Beta: {str(beta)}')
        iter_i = np.sum(root_dif * root_dif)
        ll = likelihood(Y, pi)
    return beta


# Comprobacion experimental
X = np.array(range(10)).reshape(10,1)
# print(X)
Y = [0,0,0,0,1,0,1,0,1,1]
bias = np.ones(10).reshape(10,1)
X_new = np.append(X, bias, axis=1)
# print(X_new)
a = logistics(X, Y, 0.00001)
print(a)

# Con el paquete de statsmodel de python
logit_model = sm.Logit(Y,X_new)
result = logit_model.fit()
print(result.summary2())