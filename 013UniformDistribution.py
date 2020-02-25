import numpy as np
import matplotlib.pyplot as plt

a = 1
b = 100
n = 100000
data = np.random.uniform(a, b, n)

# plot1 = plt.hist(data, bins=5)
plot2 = plt.hist(data)
# plot3 = plt.hist(data, bins=20)
# plot4 = plt.hist(data, bins=50)
plt.show()