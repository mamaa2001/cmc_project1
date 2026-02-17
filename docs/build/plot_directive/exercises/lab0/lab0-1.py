import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0.0, 10.0, 100)
f_x = np.sin(x - 2)*np.exp(-x**2)
plt.figure()
plt.plot(x, f_x)
plt.title("Lab 0 : Plotting exercise")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid()
plt.show()