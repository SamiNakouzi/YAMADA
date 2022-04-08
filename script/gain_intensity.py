import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.integrate import odeint
from sim import yamada_model

t = np.linspace(0, 2000, 4000)
dt = [80, 130, 500, 600]
netgain = []
Intensity = []
G=[]
model = yamada_model(mu1 = 2.43)

for i in range(len(dt)):
    model.integrate(t, dt[i], 0.7) #(time, duration, perturbation amplitude)
    x = model.sol.tolist()
    I = []
    ng = []
    g=[]
    for j in range(0, 4000):
        ng.append(x[j][0] - (x[j][1] - 1))
        #g.append(x[j][0])
        I.append(x[j][2])
    netgain.append(ng)
    Intensity.append(I)
    #G.append(g)
    
    
#PLOTTING:
fig, axs = plt.subplots(4, 2, sharex = True)
axs[0, 0].plot(t, netgain[0],color = 'r', label = '$\Delta t = 5$')
axs[0, 0].text(1500, 1.6, '$\Delta t = 80$', fontsize = 10)
axs[0, 1].plot(t, Intensity[0],color = 'r')

axs[1, 0].plot(t, netgain[1],color = 'green', label = '$\Delta t = 20$')
axs[1, 1].plot(t, Intensity[1], color = 'green')
axs[1, 0].text(1500, 1.65, '$\Delta t = 130$', fontsize = 10)

axs[2, 0].plot(t, netgain[2],color = 'blue', label = '$\Delta t = 500$')
axs[2, 1].plot(t, Intensity[2],color = 'blue')
axs[2, 0].text(1500, 2.5, '$\Delta t = 500$', fontsize = 10)

axs[3, 0].plot(t, netgain[3],color = 'orange', label = '$\Delta t = 600$')
axs[3, 1].plot(t, Intensity[3],color = 'orange')
axs[3, 0].text(1500, 2.5, '$\Delta t = 600$', fontsize = 10)
plt.xlabel("Time (u.arb)")
fig.suptitle('blabla')
plt.show()
