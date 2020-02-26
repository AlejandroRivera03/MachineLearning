import numpy as np
import matplotlib.pyplot as plt

# Generando valores aleatorios para la normal (media=0, desviacion tipica=1)
data = np.random.randn(10000)

x = range(1,10001)
plt.plot(x, data)
plt.show()

plt.hist(data)
plt.show()

# Funcion de distribucion acumulada
plt.plot(x, sorted(data))
plt.show()

# Changing the media and the steps
media = 5.5
sd = 2.5 #
z_10000 = np.random.randn(10000)
data = media + sd * z_10000
plt.hist(data)
plt.show()

data = np.random.randn(2,4)
print(data)