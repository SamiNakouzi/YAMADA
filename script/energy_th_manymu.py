######### This script is the same as the energy th.py but for many inital pumps ########
import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.integrate import odeint
from sim import yamada_model

inv_dmu = [] #inverse of the perturbation matrix
temp = [] #time matrix
set_inv_dmu = [] #set of inv_dmu for different pumps
set_temp = [] #set of time matricies for different pumps

t = np.linspace(300, 600, 1000)
dt = np.linspace(50 , 400, 16)

for pump in [2.43]: 
    model = yamada_model(mu1 = pump)
    inv_dmu = []
    temp = []
    for i in range(len(dt)):
        for p in np.arange(0, 1.1 ,0.001):
            model.perturbate(dt[i], p)
            model.integrate(t) 
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
    set_inv_dmu.append(inv_dmu)
    set_temp.append(temp)


#Ploting:

style.use("seaborn")
plt.plot(set_temp[0], set_inv_dmu[0], color = 'r', label = 'mu = 2.43')
#plt.plot(set_temp[1], set_inv_dmu[1], color = 'orange', label = 'mu = 2.5')
#plt.plot(set_temp[2], set_inv_dmu[2], color = 'yellow', label = 'mu = 2.55')
#plt.plot(set_temp[3], set_inv_dmu[3], color = 'limegreen', label = 'mu = 2.6')
#plt.plot(set_temp[4], set_inv_dmu[4], color = 'skyblue', label = 'mu = 2.65')
plt.xlabel('$\Delta t$')
plt.ylabel('$\dfrac{1}{\delta \mu}$', rotation=0, fontsize=12, labelpad=20)
plt.text(100, max(inv_dmu) + 0.3, "$\mu1 = 2.43$")
plt.title('$\dfrac{1}{\delta \mu} Vs \Delta t$')
plt.show()
