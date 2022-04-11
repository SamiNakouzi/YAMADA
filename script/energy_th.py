import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.integrate import odeint
from sim import yamada_model




inv_dmu = []
temp = []

t = np.linspace(300, 600, 1000)
dt = np.linspace(50 , 800, 24)

model = yamada_model()

for i in range(len(dt)):
    for p in np.arange(0, 1.1 ,0.001):
        model.integrate(t, dt[i], p) #(time, duration, perturbation amplitude)
        x = model.sol.tolist()
        I = []
        for j in range(0, len(t)):
            I.append(x[j][2])
        if (max(I)) >= 10:
            inv_dmu.append(p**(-1))
            temp.append(dt[i])
            break
        else:
            continue
        
        

for i in range(len(temp)):
    if inv_dmu[i] == inv_dmu[i+1]:
        threashold = temp[i]
        break
#Ploting:

style.use("seaborn")
plt.scatter(temp, inv_dmu,color = 'r', label = "dots")
plt.vlines(threashold, color = 'blue',ymin = min(inv_dmu), ymax = max(inv_dmu), ls = '--')
plt.plot(temp, inv_dmu, '--k', alpha = 0.5)
plt.xlabel('$\Delta t$')
plt.ylabel('$\dfrac{1}{\delta \mu}$', rotation=0, fontsize=12, labelpad=20)
plt.text(threashold, min(inv_dmu)-0.1, '$\Delta t = 278.26$')
#plt.legend()
plt.title('$\dfrac{1}{\delta \mu} Vs \Delta t$')
plt.show()
