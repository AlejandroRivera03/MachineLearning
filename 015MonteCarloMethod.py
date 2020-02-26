# Generamos dos numeros aleatorios uniforme x e y entre 0 y 1 en total 1000 veces.
# Calculamos x*x + y*y.
#   Si el valor es inferior a 1 => estamos dentro del circulo.
#   Si el valor es superior a 1 => estamos fuera del circulo.
# Calculamos el numero total de veces que estan dentro del circulo y lo dividimos entre el numero   
# total de intentos para obtener una aproximacion de la probabilidad de caer dentro del circulo.
# Usamos dicha probabilidad para aproxmar el valor de pi
# Repetimos el experimento un numero suficiente de veces (por ejemplo 100), para obtener (100) direfentes aproximacionde pi
# Calculamos el promedio de los 100 experimentos anteriores para dar un valor final de pi

import numpy as np
import matplotlib.pyplot as plt

def pi_montecarlo(n, n_exp):
    pi_avg = 0
    pi_value_list = []
    for i in range(n_exp):
        value = 0
        x = np.random.uniform(0, 1, n).tolist()
        y = np.random.uniform(0, 1, n).tolist()
        for j in range(n):
            z = np.sqrt(x[j]*x[j] + y[j]*y[j])
            if(z<=1):
                value += 1
        float_value = float(value)
        pi_value = float_value * 4 / n
        pi_value_list.append(pi_value)
        pi_avg += pi_value
    pi = pi_avg/n_exp

    print(pi)
    fig = plt.plot(pi_value_list)
    return (pi, fig)

pi_montecarlo(10000, 200)
plt.show()